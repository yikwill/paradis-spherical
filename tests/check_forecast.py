"""Exercise native forecasting on real ERA5, including changed horizon and chunking."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
import xarray as xr

from data.datamodule import Era5DataModule
from experiments.weather import weather_config
from forecast import predict_batch
from trainer import ParadisTask, configure_optimizers, save_checkpoint, load_checkpoint
from utils.file_output import ZarrForecastWriter


def main():
    torch.set_num_threads(2)
    root=sys.argv[1] if len(sys.argv)>1 else '/home/yikwill/paradis-data'
    cfg=weather_config(root,'disco',20,2)
    cfg.forecast.enable=True
    cfg.forecast.start_date='2017-01-01T06:00:00'
    cfg.forecast.end_date='2017-01-01T12:00:00'
    cfg.forecast.write_every_n=1
    cfg.forecast.output_frequency=2
    data=Era5DataModule(cfg)
    data.setup('predict')
    task=ParadisTask(data,cfg).eval()
    opt,scheduler=configure_optimizers(task.model,cfg,20)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    with TemporaryDirectory() as directory:
        checkpoint=Path(directory)/'model.pt'
        save_checkpoint(checkpoint,task,opt,scheduler,scaler,dict(step=0))
        cfg.model.forecast_steps=3
        data=Era5DataModule(cfg)
        data.setup('predict')
        task=ParadisTask(data,cfg).eval()
        load_checkpoint(checkpoint,task)
        cfg.forecast.output_file=str(Path(directory)/'forecast.zarr')
        writer=ZarrForecastWriter(cfg,data.dataset)
        writer.init_store(data.dataset)
        with torch.inference_mode():
            for batch in data.predict_dataloader():
                predict_batch(task,batch,writer)
        output=xr.open_zarr(cfg.forecast.output_file,consolidated=True)
        np.testing.assert_array_equal(output.prediction_timedelta.values,np.array([0,6,18],dtype='timedelta64[h]'))
        assert output.sizes['time']==2 and output.sizes['prediction_timedelta']==3
        assert np.isfinite(output['2m_temperature'].values).all()
        assert np.isfinite(output.geopotential.values).all()
        try:
            ZarrForecastWriter(cfg,data.dataset).init_store(data.dataset)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing forecast data was overwritten')
    print('PASS: real-data native forecast, checkpoint horizon change, incremental Zarr output, lead times, overwrite protection')


if __name__=='__main__':
    main()
