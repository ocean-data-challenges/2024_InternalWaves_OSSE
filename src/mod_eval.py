import xarray as xr
import numpy
import logging
import xrft
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")
import xscale
import xscale.xscale.spectral.fft as xfft
from matplotlib.colors import LogNorm 
import matplotlib.pylab as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import Wavenum_freq_spec_func as wfs

def rmse_based_scores(ds_oi, ds_ref, var_list=['sossheig']):
    
    rmse_t_list = dict()
    rmse_xy_list = dict()
    leaderboard_rmse_list = dict()
    reconstruction_error_stability_metric_list = dict()
    for var in var_list:
    
        logging.info('     Compute RMSE-based scores for ...'+ var)

        # RMSE(t) based score
        rmse_t = 1.0 - (((ds_oi[var] - ds_ref[var])**2).mean(dim=('lon', 'lat')))**0.5/(((ds_ref[var])**2).mean(dim=('lon', 'lat')))**0.5
        # RMSE(x, y) based score
        # rmse_xy = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
        rmse_xy = (((ds_oi[var] - ds_ref[var])**2).mean(dim=('time')))**0.5

        rmse_t = rmse_t.rename('rmse_t')
        rmse_xy = rmse_xy.rename('rmse_xy')

        # Temporal stability of the error
        reconstruction_error_stability_metric = rmse_t.std().values

        # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
        leaderboard_rmse = 1.0 - (((ds_oi[var] - ds_ref[var]) ** 2).mean()) ** 0.5 / (
            ((ds_ref[var]) ** 2).mean()) ** 0.5

        logging.info('          => Leaderboard SSH RMSE score = %s', numpy.round(leaderboard_rmse.values, 2))
        logging.info('          Error variability = %s (temporal stability of the mapping error)', numpy.round(reconstruction_error_stability_metric, 2))
        
        
        rmse_t_list.update({var:rmse_t})
        rmse_xy_list.update({var:rmse_xy})
        leaderboard_rmse_list.update({var:numpy.round(leaderboard_rmse.values, 2)})
        reconstruction_error_stability_metric_list.update({var:numpy.round(reconstruction_error_stability_metric, 2)})
        
    
    return rmse_t_list, rmse_xy_list, leaderboard_rmse_list, reconstruction_error_stability_metric_list


def psd_based_scores(ds_oi, ds_ref, var_list=['sossheig']):
    
    nsr_score_list = dict()
    eff_res_spat_list = dict()
    eff_res_temp_list = dict()
    
    for var in var_list:
    
        logging.info('     Compute PSD-based scores for ...'+ var)
 
        # Compute error = SSH_reconstruction - SSH_true
        err = (ds_oi[var] - ds_ref[var])
        err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
        # make time vector in days units 
        err['time'] = (err.time - err.time[0]) / numpy.timedelta64(1, 'D')

        # Rechunk SSH_true
        signal = ds_ref[var].chunk({"lat":1, 'time': ds_ref['time'].size, 'lon': ds_ref['lon'].size})
        # make time vector in days units
        signal['time'] = (signal.time - signal.time[0]) / numpy.timedelta64(1, 'D')

        # Compute PSD_err and PSD_signal
        psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='constant', window=True).compute()
        psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='constant', window=True).compute()

        # Averaged over latitude
        mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
        mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)

        # return PSD-based score
        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score



        level = [0.5]
        cs = plt.contour(1./psd_based_score.freq_lon.values,1./psd_based_score.freq_time.values, psd_based_score, level)
        x05, y05 = cs.collections[0].get_paths()[0].vertices.T
        plt.close()

        shortest_spatial_wavelength_resolved = numpy.min(x05)
        shortest_temporal_wavelength_resolved = numpy.min(y05)

        logging.info('          => Leaderboard Spectral score = %s (degree lon)',
                     numpy.round(shortest_spatial_wavelength_resolved, 2))
        logging.info('          => shortest temporal wavelength resolved = %s (days)',
                     numpy.round(shortest_temporal_wavelength_resolved, 2))
        
        nsr_score_list.update({var:1.0 - mean_psd_err/mean_psd_signal})
        eff_res_spat_list.update({var:numpy.round(shortest_spatial_wavelength_resolved, 2)})
        eff_res_temp_list.update({var:numpy.round(shortest_temporal_wavelength_resolved, 2)})

    return nsr_score_list, eff_res_spat_list, eff_res_temp_list




def plot_omegak_spectrum(ds_ref, ds_maps, figsave=None):

    lon, lat = numpy.meshgrid(ds_maps.lon,ds_maps.lat)  

    ds_maps = ds_maps.rename_dims({'lon':'y','lat':'x'})
    ds_maps = ds_maps.assign_coords({'nav_lon':(['y','x'],lon)})
    ds_maps = ds_maps.assign_coords({'nav_lat':(['y','x'],lat)})
    ds_maps = ds_maps.rename({'time':'time_counter'}) 
    
    Mapsbox=ds_maps['SSH_tot'] 

    dx,dy = wfs.get_dx_dy(Mapsbox[0])
    Maps_No_NaN = Mapsbox.interpolate_na(dim='y')
    Maps_dtr = wfs.detrendn(Maps_No_NaN,axes=[0,1,2])
    Maps_wdw = wfs.apply_window(Maps_dtr, Maps_dtr.dims, window_type='hanning')
    Mapshat = xfft.fft(Maps_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps_psd = xfft.psd(Mapshat)
    Maps_frequency,kx,ky = wfs.get_f_kx_ky(Mapshat)
    Maps_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps_psd_np = Maps_psd.values
    Maps_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps_wavenumber,Maps_psd_np)
    
    Maps1box=ds_maps['ssh_it'] 

    dx,dy = wfs.get_dx_dy(Maps1box[0])
    Maps1_No_NaN = Maps1box.interpolate_na(dim='y')
    Maps1_dtr = wfs.detrendn(Maps1_No_NaN,axes=[0,1,2])
    Maps1_wdw = wfs.apply_window(Maps1_dtr, Maps1_dtr.dims, window_type='hanning')
    Maps1hat = xfft.fft(Maps1_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps1_psd = xfft.psd(Maps1hat)
    Maps1_frequency,kx,ky = wfs.get_f_kx_ky(Maps1hat)
    Maps1_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps1_psd_np = Maps1_psd.values
    Maps1_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps1_wavenumber,Maps1_psd_np)

    Maps2box=ds_maps['ssh_bm'] 

    dx,dy = wfs.get_dx_dy(Maps2box[0])
    Maps2_No_NaN = Maps2box.interpolate_na(dim='y')
    Maps2_dtr = wfs.detrendn(Maps2_No_NaN,axes=[0,1,2])
    Maps2_wdw = wfs.apply_window(Maps2_dtr, Maps2_dtr.dims, window_type='hanning')
    Maps2hat = xfft.fft(Maps2_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps2_psd = xfft.psd(Maps2hat)
    Maps2_frequency,kx,ky = wfs.get_f_kx_ky(Maps2hat)
    Maps2_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps2_psd_np = Maps2_psd.values
    Maps2_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps2_wavenumber,Maps2_psd_np)
    
    
    lon, lat = numpy.meshgrid(ds_ref.lon,ds_ref.lat)  

    ds_ref = ds_ref.rename_dims({'lon':'y','lat':'x'})
    ds_ref = ds_ref.assign_coords({'nav_lon':(['y','x'],lon)})
    ds_ref = ds_ref.assign_coords({'nav_lat':(['y','x'],lat)})
    ds_ref = ds_ref.rename({'time':'time_counter'})
    
    Refbox=ds_ref['ssh'] 

    dx,dy = wfs.get_dx_dy(Refbox[0])
    Ref_No_NaN = Refbox.interpolate_na(dim='y')
    Ref_dtr = wfs.detrendn(Ref_No_NaN,axes=[0,1,2])
    Ref_wdw = wfs.apply_window(Ref_dtr, Ref_dtr.dims, window_type='hanning')
    Refhat = xfft.fft(Ref_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref_psd = xfft.psd(Refhat)
    Ref_frequency,kx,ky = wfs.get_f_kx_ky(Refhat)
    Ref_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref_psd_np = Ref_psd.values
    Ref_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref_wavenumber,Ref_psd_np)
    
    Ref1box=ds_ref['ssh_it'] 

    dx,dy = wfs.get_dx_dy(Ref1box[0])
    Ref1_No_NaN = Ref1box.interpolate_na(dim='y')
    Ref1_dtr = wfs.detrendn(Ref1_No_NaN,axes=[0,1,2])
    Ref1_wdw = wfs.apply_window(Ref1_dtr, Ref1_dtr.dims, window_type='hanning')
    Ref1hat = xfft.fft(Ref1_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref1_psd = xfft.psd(Ref1hat)
    Ref1_frequency,kx,ky = wfs.get_f_kx_ky(Ref1hat)
    Ref1_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref1_psd_np = Ref1_psd.values
    Ref1_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref1_wavenumber,Ref1_psd_np)

    Ref2box=ds_ref['ssh_bm'] 

    dx,dy = wfs.get_dx_dy(Ref2box[0])
    Ref2_No_NaN = Ref2box.interpolate_na(dim='y')
    Ref2_dtr = wfs.detrendn(Ref2_No_NaN,axes=[0,1,2])
    Ref2_wdw = wfs.apply_window(Ref2_dtr, Ref2_dtr.dims, window_type='hanning')
    Ref2hat = xfft.fft(Ref2_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref2_psd = xfft.psd(Ref2hat)
    Ref2_frequency,kx,ky = wfs.get_f_kx_ky(Ref2hat)
    Ref2_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref2_psd_np = Ref2_psd.values
    Ref2_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref2_wavenumber,Ref2_psd_np)
    
    sec_to_hour = 3600.0
    T=2*numpy.pi/(1E-4)
    norm = LogNorm(vmin=0.001,vmax=1000)
    cmap = 'seismic'

    fig=plt.figure(figsize=(20,30))


    ax = plt.subplot(321)
    plt.pcolormesh(Ref_wavenumber,sec_to_hour*Ref_frequency,Ref_wavenum_freq_spectrum,norm=norm,cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref_wavenumber.min(),Ref_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('Spectrum SSH reference tot',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    ax = plt.subplot(322)
    plt.pcolormesh(Maps_wavenumber,sec_to_hour*Maps_frequency,Maps_wavenum_freq_spectrum,norm=norm,cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Maps_wavenumber.min(),Maps_wavenumber.max())
    #ax.set_ylim(1E-5,5E-1)
    plt.axhline(y=1/T,color='k', linewidth=0)
    ax.set_title('Spectrum SSH reconstructed',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    ax = plt.subplot(323)
    plt.pcolormesh(Ref1_wavenumber,sec_to_hour*Ref1_frequency,Ref1_wavenum_freq_spectrum,norm=norm,cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref1_wavenumber.min(),Ref1_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('Spectrum SSH reference IT',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()

    ax = plt.subplot(324)
    plt.pcolormesh(Maps1_wavenumber,sec_to_hour*Maps1_frequency,Maps1_wavenum_freq_spectrum,norm=norm,cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Maps1_wavenumber.min(),Maps1_wavenumber.max())
    #ax.set_ylim(1E-5,5E-1)
    plt.axhline(y=1/T,color='k', linewidth=0)
    ax.set_title('Spectrum SSH-IT',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    ax = plt.subplot(325)
    plt.pcolormesh(Ref2_wavenumber,sec_to_hour*Ref2_frequency,Ref2_wavenum_freq_spectrum,norm=norm,cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref2_wavenumber.min(),Ref2_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('Spectrum SSH reference BM',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    ax = plt.subplot(326)
    plt.pcolormesh(Maps2_wavenumber,sec_to_hour*Maps2_frequency,Maps2_wavenum_freq_spectrum,norm=norm,cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Maps2_wavenumber.min(),Maps2_wavenumber.max())
    #ax.set_ylim(1E-5,5E-1)
    plt.axhline(y=1/T,color='k', linewidth=0)
    ax.set_title('Spectrum SSH-BM',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()
    #ax.text(-0.32, 0.8, "Frequency cutoff", transform=ax.transAxes,color='g',size=12)
    #ax.text(-0.32, 0.8, " ", transform=ax.transAxes,color='g',size=12)

    if figsave is not None: 
        fig.savefig(figsave, facecolor='w')
        
        
def plot_nsr_spectrum(ds_ref, ds_maps, figsave=None):
    
    
    lon, lat = numpy.meshgrid(ds_ref.lon,ds_ref.lat)  

    ds_ref = ds_ref.rename_dims({'lon':'y','lat':'x'})
    ds_ref = ds_ref.assign_coords({'nav_lon':(['y','x'],lon)})
    ds_ref = ds_ref.assign_coords({'nav_lat':(['y','x'],lat)})
    ds_ref = ds_ref.rename({'time':'time_counter'})
    
    Ref0box=ds_ref['ssh_it']+ds_ref['ssh_bm']

    dx,dy = wfs.get_dx_dy(Ref0box[0])
    Ref0_No_NaN = Ref0box.interpolate_na(dim='y')
    Ref0_dtr = wfs.detrendn(Ref0_No_NaN,axes=[0,1,2])
    Ref0_wdw = wfs.apply_window(Ref0_dtr, Ref0_dtr.dims, window_type='hanning')
    Ref0hat = xfft.fft(Ref0_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref0_psd = xfft.psd(Ref0hat)
    Ref0_frequency,kx,ky = wfs.get_f_kx_ky(Ref0hat)
    Ref0_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref0_psd_np = Ref0_psd.values
    Ref0_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref0_wavenumber,Ref0_psd_np)
    
    
    Refbox=ds_ref['ssh'] 

    dx,dy = wfs.get_dx_dy(Refbox[0])
    Ref_No_NaN = Refbox.interpolate_na(dim='y')
    Ref_dtr = wfs.detrendn(Ref_No_NaN,axes=[0,1,2])
    Ref_wdw = wfs.apply_window(Ref_dtr, Ref_dtr.dims, window_type='hanning')
    Refhat = xfft.fft(Ref_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref_psd = xfft.psd(Refhat)
    Ref_frequency,kx,ky = wfs.get_f_kx_ky(Refhat)
    Ref_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref_psd_np = Ref_psd.values
    Ref_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref_wavenumber,Ref_psd_np)
    
    Ref1box=ds_ref['ssh_it'] 

    dx,dy = wfs.get_dx_dy(Ref1box[0])
    Ref1_No_NaN = Ref1box.interpolate_na(dim='y')
    Ref1_dtr = wfs.detrendn(Ref1_No_NaN,axes=[0,1,2])
    Ref1_wdw = wfs.apply_window(Ref1_dtr, Ref1_dtr.dims, window_type='hanning')
    Ref1hat = xfft.fft(Ref1_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref1_psd = xfft.psd(Ref1hat)
    Ref1_frequency,kx,ky = wfs.get_f_kx_ky(Ref1hat)
    Ref1_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref1_psd_np = Ref1_psd.values
    Ref1_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref1_wavenumber,Ref1_psd_np)

    Ref2box=ds_ref['ssh_bm'] 

    dx,dy = wfs.get_dx_dy(Ref2box[0])
    Ref2_No_NaN = Ref2box.interpolate_na(dim='y')
    Ref2_dtr = wfs.detrendn(Ref2_No_NaN,axes=[0,1,2])
    Ref2_wdw = wfs.apply_window(Ref2_dtr, Ref2_dtr.dims, window_type='hanning')
    Ref2hat = xfft.fft(Ref2_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Ref2_psd = xfft.psd(Ref2hat)
    Ref2_frequency,kx,ky = wfs.get_f_kx_ky(Ref2hat)
    Ref2_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Ref2_psd_np = Ref2_psd.values
    Ref2_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Ref2_wavenumber,Ref2_psd_np)
    
    
    
    lon, lat = numpy.meshgrid(ds_maps.lon,ds_maps.lat)  

    ds_maps = ds_maps.rename_dims({'lon':'y','lat':'x'})
    ds_maps = ds_maps.assign_coords({'nav_lon':(['y','x'],lon)})
    ds_maps = ds_maps.assign_coords({'nav_lat':(['y','x'],lat)})
    ds_maps = ds_maps.rename({'time':'time_counter'}) 
    
    Maps0box=ds_maps['ssh'] - Ref0box

    dx,dy = wfs.get_dx_dy(Maps0box[0])
    Maps0_No_NaN = Maps0box.interpolate_na(dim='y')
    Maps0_dtr = wfs.detrendn(Maps0_No_NaN,axes=[0,1,2])
    Maps0_wdw = wfs.apply_window(Maps0_dtr, Maps0_dtr.dims, window_type='hanning')
    Maps0hat = xfft.fft(Maps0_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps0_psd = xfft.psd(Maps0hat)
    Maps0_frequency,kx,ky = wfs.get_f_kx_ky(Maps0hat)
    Maps0_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps0_psd_np = Maps0_psd.values
    Maps0_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps0_wavenumber,Maps0_psd_np)
    
    Mapsbox=ds_maps['ssh'] - Refbox

    dx,dy = wfs.get_dx_dy(Mapsbox[0])
    Maps_No_NaN = Mapsbox.interpolate_na(dim='y')
    Maps_dtr = wfs.detrendn(Maps_No_NaN,axes=[0,1,2])
    Maps_wdw = wfs.apply_window(Maps_dtr, Maps_dtr.dims, window_type='hanning')
    Mapshat = xfft.fft(Maps_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps_psd = xfft.psd(Mapshat)
    Maps_frequency,kx,ky = wfs.get_f_kx_ky(Mapshat)
    Maps_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps_psd_np = Maps_psd.values
    Maps_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps_wavenumber,Maps_psd_np)
    
    Maps1box=ds_maps['ssh_it'] - Ref1box

    dx,dy = wfs.get_dx_dy(Maps1box[0])
    Maps1_No_NaN = Maps1box.interpolate_na(dim='y')
    Maps1_dtr = wfs.detrendn(Maps1_No_NaN,axes=[0,1,2])
    Maps1_wdw = wfs.apply_window(Maps1_dtr, Maps1_dtr.dims, window_type='hanning')
    Maps1hat = xfft.fft(Maps1_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps1_psd = xfft.psd(Maps1hat)
    Maps1_frequency,kx,ky = wfs.get_f_kx_ky(Maps1hat)
    Maps1_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps1_psd_np = Maps1_psd.values
    Maps1_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps1_wavenumber,Maps1_psd_np)

    Maps2box=ds_maps['ssh_bm'] - Ref2box

    dx,dy = wfs.get_dx_dy(Maps2box[0])
    Maps2_No_NaN = Maps2box.interpolate_na(dim='y')
    Maps2_dtr = wfs.detrendn(Maps2_No_NaN,axes=[0,1,2])
    Maps2_wdw = wfs.apply_window(Maps2_dtr, Maps2_dtr.dims, window_type='hanning')
    Maps2hat = xfft.fft(Maps2_wdw, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
    Maps2_psd = xfft.psd(Maps2hat)
    Maps2_frequency,kx,ky = wfs.get_f_kx_ky(Maps2hat)
    Maps2_wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)
    Maps2_psd_np = Maps2_psd.values
    Maps2_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,Maps2_wavenumber,Maps2_psd_np)
    
    sec_to_hour = 3600.0
    T=2*numpy.pi/(1E-4) 
    cmap = 'seismic'


    fig=plt.figure(figsize=(20,20))



    ax = plt.subplot(221)
    plt.pcolormesh(Ref0_wavenumber,sec_to_hour*Ref0_frequency,1-Maps0_wavenum_freq_spectrum/Ref0_wavenum_freq_spectrum,vmin=0.,vmax=1,cmap='inferno')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref0_wavenumber.min(),Ref0_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('NSR spectrum SSH_tot',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()
     
    ax = plt.subplot(222)
    plt.pcolormesh(Ref_wavenumber,sec_to_hour*Ref_frequency,1-Maps_wavenum_freq_spectrum/Ref_wavenum_freq_spectrum,vmin=0.,vmax=1,cmap='inferno')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref_wavenumber.min(),Ref_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('NSR spectrum SSH_BM-IT',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    ax = plt.subplot(223)
    plt.pcolormesh(Ref1_wavenumber,sec_to_hour*Ref1_frequency,1-Maps1_wavenum_freq_spectrum/Ref1_wavenum_freq_spectrum,vmin=0.,vmax=1,cmap='inferno')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref1_wavenumber.min(),Ref1_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('NSR spectrum SSH_IT',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    ax = plt.subplot(224)
    plt.pcolormesh(Ref2_wavenumber,sec_to_hour*Ref2_frequency,1-Maps2_wavenum_freq_spectrum/Ref2_wavenum_freq_spectrum,vmin=0.,vmax=1,cmap='inferno')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
    ax.set_ylabel('frequency (cph)',fontsize=15)
    ax.set_xlim(Ref2_wavenumber.min(),Ref2_wavenumber.max())
    #ax.set_ylim(1E-5,8E-1)
    plt.axhline(y=1/T,color='g', linewidth=0)
    ax.set_title('NSR spectrum SSH_BM',size=18)
    ax.tick_params(labelsize=15)
    plt.colorbar()


    if figsave is not None: 
        fig.savefig(figsave, facecolor='w')
        
        
        
        
        
def plot_temporal_spectrum(dc_ref,ds_reconstruction_regrid,figsave=None):

    import scipy

    npt = numpy.shape(dc_ref.ssh)[0]
    fs = 1 # 1 hour

    ssh_tmp = numpy.array(dc_ref.ssh) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_ref_tot, psd_ref_tot = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)

    ssh_tmp = numpy.array(dc_ref.ssh_bm) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_ref_bm, psd_ref_bm = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)

    ssh_tmp = numpy.array(dc_ref.ssh_bm+dc_ref.ssh_it) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_ref_bmit, psd_ref_bmit = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)

    ssh_tmp = numpy.array(dc_ref.ssh_it) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_ref_it, psd_ref_it = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)



    ssh_tmp = numpy.array(ds_reconstruction_regrid.ssh) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_rec_tot, psd_rec_tot = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)

    ssh_tmp = numpy.array(ds_reconstruction_regrid.ssh_bm) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_rec_bm, psd_rec_bm = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)

    ssh_tmp = numpy.array(ds_reconstruction_regrid.ssh_it) 
    ssh_tmp = ssh_tmp[:,~numpy.isnan(ssh_tmp).any(axis=0)]
    ssh_tmp = numpy.ma.masked_invalid(numpy.ravel(ssh_tmp,'F'))

    wn_rec_it, psd_rec_it = scipy.signal.welch(ssh_tmp,
                                                             fs=fs,
                                                             nperseg=npt,
                                                             scaling='density',
                                                             noverlap=0)
    
    
    
    plt.figure(figsize=(20,10))

    plt.subplot(221)
    plt.title('SSH_tot',fontsize=16)
    plt.loglog(wn_ref_tot,psd_ref_tot,label='Ref')
    plt.loglog(wn_rec_tot,psd_rec_tot,label='VarDyn')

    ticks = [1/5,1/10,1/24,1/48,1/72,1/120,1/240]
    ticks_lab = ['5h','10h','1d','2d','3d','5d','10d']
    plt.yticks(fontsize=14)
    plt.xticks(ticks,ticks_lab,fontsize=14)
    plt.axis([1/400, 1/2, 1e-5, 1])
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.subplot(222)
    plt.title('SSH_BM+IT',fontsize=16)
    plt.loglog(wn_ref_bmit,psd_ref_bmit,label='Ref')
    plt.loglog(wn_rec_tot,psd_rec_tot,label='VarDyn')

    ticks = [1/5,1/10,1/24,1/48,1/72,1/120,1/240]
    ticks_lab = ['5h','10h','1d','2d','3d','5d','10d']
    plt.yticks(fontsize=14)
    plt.xticks(ticks,ticks_lab,fontsize=14)
    plt.axis([1/400, 1/2, 1e-5, 1])
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.subplot(223)
    plt.title('SSH_BM',fontsize=16)
    plt.loglog(wn_ref_bm,psd_ref_bm,label='Ref')
    plt.loglog(wn_rec_bm,psd_rec_bm,label='VarDyn')

    ticks = [1/5,1/10,1/24,1/48,1/72,1/120,1/240]
    ticks_lab = ['5h','10h','1d','2d','3d','5d','10d']
    plt.yticks(fontsize=14)
    plt.xticks(ticks,ticks_lab,fontsize=14)
    plt.axis([1/400, 1/2, 1e-5, 1])
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.subplot(224)
    plt.title('SSH_IT',fontsize=16)
    plt.loglog(wn_ref_it,psd_ref_it,label='Ref')
    plt.loglog(wn_rec_it,psd_rec_it,label='VarDyn')
    #plt.loglog([f,f],[1E-13,1],'k--')

    ticks = [1/5,1/10,1/24,1/48,1/72,1/120,1/240]
    ticks_lab = ['5h','10h','1d','2d','3d','5d','10d']
    plt.yticks(fontsize=14)
    plt.xticks(ticks,ticks_lab,fontsize=14)
    plt.axis([1/400, 1/2, 1e-5, 1])
    plt.grid(True)
    plt.legend(fontsize=14)
    if figsave is not None: 
        plt.savefig(figsave, facecolor='w')
    plt.show()
    
    
    
        
        
    plt.figure(figsize=(15,10))
        
    plt.title('SSH',fontsize=16)
    plt.loglog(wn_ref_tot,psd_ref_tot,'k--',label='Ref-Tot') 
    plt.loglog(wn_ref_bm,psd_ref_bm,'b--',label='Ref-BM')
    plt.loglog(wn_ref_it,psd_ref_it,'r--',label='Ref-IT')
    plt.loglog(wn_ref_tot,psd_rec_tot,'k',label='Rec-Tot') 
    plt.loglog(wn_ref_bm,psd_rec_bm,'b',label='Rec-BM')
    plt.loglog(wn_ref_it,psd_rec_it,'r',label='Rec-IT')

    ticks = [1/5,1/10,1/24,1/48,1/72,1/120,1/240]
    ticks_lab = ['5h','10h','1d','2d','3d','5d','10d']
    plt.yticks(fontsize=14)
    plt.xticks(ticks,ticks_lab,fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.axis([1/400, 1/2, 1e-5, 1])
    if figsave is not None: 
        plt.savefig(figsave[:-4]+'_total.png', facecolor='w')
    plt.show()
 


def plot_maps_and_differences(dc_ref,ds_reconstruction_regrid,it=0,figsave=None): 
    plt.figure(figsize=(30,30))
    plt.subplot(331)
    plt.title('SSH ref',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,dc_ref.ssh[it],cmap='inferno',vmin=0.7,vmax=1.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()
    plt.subplot(332)
    plt.title('SSH reconstruction',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,ds_reconstruction_regrid.ssh[it],cmap='inferno',vmin=0.7,vmax=1.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()
    plt.subplot(333)
    plt.title('SSH difference',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,dc_ref.ssh[it]-ds_reconstruction_regrid.ssh[it],cmap='seismic',vmin=-0.4,vmax=0.4)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()

    plt.subplot(334)
    plt.title('SSH-BM ref',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,dc_ref.ssh_bm[it],cmap='cividis',vmin=0.7,vmax=1.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()
    plt.subplot(335)
    plt.title('SSH-BM reconstruction',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,ds_reconstruction_regrid.ssh_bm[it],cmap='cividis',vmin=0.7,vmax=1.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()
    plt.subplot(336)
    plt.title('SSH-BM difference',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,dc_ref.ssh_bm[it]-ds_reconstruction_regrid.ssh_bm[it],cmap='seismic',vmin=-0.4,vmax=0.4)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()

    plt.subplot(337)
    plt.title('SSH-IT ref',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,dc_ref.ssh_it[it],cmap='bwr',vmin=-0.1,vmax=0.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()
    plt.subplot(338)
    plt.title('SSH-IT reconstruction',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,ds_reconstruction_regrid.ssh_it[it],cmap='bwr',vmin=-0.1,vmax=0.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()
    plt.subplot(339)
    plt.title('SSH-IT difference',fontsize=15)
    plt.pcolormesh(dc_ref.lon,dc_ref.lat,dc_ref.ssh_it[it]-ds_reconstruction_regrid.ssh_it[it],cmap='seismic',vmin=-0.1,vmax=0.1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar()

    if figsave is not None: 
        plt.savefig(figsave, facecolor='w')
        
    plt.show()

        
    