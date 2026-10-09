// Depthwise DISCO contraction using torch-harmonics' quadrature-weighted stencil.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {
void check(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ptr,
           const at::Tensor& ker, const at::Tensor& col, const at::Tensor& val) {
    TORCH_CHECK(x.device().is_cpu() && x.is_contiguous() && x.dim() == 4);
    TORCH_CHECK(w.device().is_cpu() && w.is_contiguous() && w.dim() == 3);
    TORCH_CHECK(w.size(0) == x.size(1) && w.size(1) == 1 && w.scalar_type() == x.scalar_type());
    TORCH_CHECK(ptr.device().is_cpu() && ker.device().is_cpu() && col.device().is_cpu() && val.device().is_cpu());
    TORCH_CHECK(ptr.is_contiguous() && ker.is_contiguous() && col.is_contiguous() && val.is_contiguous());
    TORCH_CHECK(ptr.scalar_type() == at::kLong && ker.scalar_type() == at::kLong && col.scalar_type() == at::kLong);
    TORCH_CHECK(ptr.numel() == x.size(2)+1 && ker.numel() == col.numel() && col.numel() == val.numel());
    TORCH_CHECK(val.scalar_type() == x.scalar_type());
}

at::Tensor forward(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& ptr,
                   const at::Tensor& kernel, const at::Tensor& column, const at::Tensor& values) {
    check(x, weight, ptr, kernel, column, values);
    auto out = at::zeros_like(x);
    const auto B=x.size(0), C=x.size(1), H=x.size(2), W=x.size(3), K=weight.size(2);
    const auto *rp=ptr.data_ptr<int64_t>(), *ki=kernel.data_ptr<int64_t>(), *ci=column.data_ptr<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "paradis_disco_forward", [&] {
        const auto *xp=x.data_ptr<scalar_t>(), *wp=weight.data_ptr<scalar_t>(), *vp=values.data_ptr<scalar_t>();
        auto *yp=out.data_ptr<scalar_t>();
        at::parallel_for(0, B*C*H, 1, [&](int64_t begin, int64_t end) {
            for (int64_t i=begin; i<end; ++i) {
                const auto h=i%H, bc=i/H, c=bc%C;
                auto *y=yp+i*W;
                for (auto e=rp[h]; e<rp[h+1];) {
                    auto end=e+1;
                    while (end<rp[h+1] && ci[end]==ci[e]) ++end;
                    const auto hi=ci[e]/W, shift=ci[e]%W;
                    scalar_t coefficient=0;
                    for (auto j=e; j<end; ++j) coefficient += vp[j]*wp[c*K+ki[j]];
                    const auto *source=xp+(bc*H+hi)*W;
                    for (int64_t j=0; j<W-shift; ++j) y[j] += coefficient*source[j+shift];
                    for (int64_t j=W-shift; j<W; ++j) y[j] += coefficient*source[j+shift-W];
                    e=end;
                }
            }
        });
    });
    return out;
}

std::tuple<at::Tensor,at::Tensor> backward(const at::Tensor& grad, const at::Tensor& x,
        const at::Tensor& weight, const at::Tensor& ptr, const at::Tensor& kernel,
        const at::Tensor& column, const at::Tensor& values) {
    check(x, weight, ptr, kernel, column, values);
    TORCH_CHECK(grad.sizes() == x.sizes() && grad.scalar_type() == x.scalar_type() && grad.device().is_cpu() && grad.is_contiguous());
    auto dx=at::zeros_like(x), dw=at::zeros_like(weight);
    const auto B=x.size(0), C=x.size(1), H=x.size(2), W=x.size(3), K=weight.size(2);
    const auto *rp=ptr.data_ptr<int64_t>(), *ki=kernel.data_ptr<int64_t>(), *ci=column.data_ptr<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "paradis_disco_backward", [&] {
        const auto *xp=x.data_ptr<scalar_t>(), *wp=weight.data_ptr<scalar_t>(), *vp=values.data_ptr<scalar_t>(), *gp=grad.data_ptr<scalar_t>();
        auto *dxp=dx.data_ptr<scalar_t>(), *dwp=dw.data_ptr<scalar_t>();
        // One worker owns all batches of a channel: no atomic input/weight sums.
        at::parallel_for(0, C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t c=begin; c<end; ++c) for (int64_t b=0; b<B; ++b) for (int64_t h=0; h<H; ++h) {
                const auto *g=gp+((b*C+c)*H+h)*W;
                for (auto e=rp[h]; e<rp[h+1];) {
                    auto end=e+1;
                    while (end<rp[h+1] && ci[end]==ci[e]) ++end;
                    const auto hi=ci[e]/W, shift=ci[e]%W;
                    scalar_t coefficient=0;
                    for (auto j=e; j<end; ++j) coefficient += vp[j]*wp[c*K+ki[j]];
                    const auto *source=xp+((b*C+c)*H+hi)*W;
                    auto *dsource=dxp+((b*C+c)*H+hi)*W;
                    scalar_t dot=0;
                    for (int64_t j=0; j<W-shift; ++j) {
                        dsource[j+shift] += coefficient*g[j];
                        dot += source[j+shift]*g[j];
                    }
                    for (int64_t j=W-shift; j<W; ++j) {
                        dsource[j+shift-W] += coefficient*g[j];
                        dot += source[j+shift-W]*g[j];
                    }
                    for (auto j=e; j<end; ++j) dwp[c*K+ki[j]] += vp[j]*dot;
                    e=end;
                }
            }
        });
    });
    return {dx,dw};
}
}

TORCH_LIBRARY(paradis_disco, m) {
    m.def("forward(Tensor x, Tensor w, Tensor ptr, Tensor kernel, Tensor column, Tensor values) -> Tensor");
    m.def("backward(Tensor grad, Tensor x, Tensor w, Tensor ptr, Tensor kernel, Tensor column, Tensor values) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(paradis_disco, CPU, m) {
    m.impl("forward", forward);
    m.impl("backward", backward);
}
