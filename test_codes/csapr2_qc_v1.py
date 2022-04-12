# -*- coding: utf-8 -*-
"""
This code is based on B. Fuchs' RELAMPAGO radar QC code (CSU CHIVO), intended to be used on CSAPR2 data. 
Uses standard deviation of phase, despeckling, NCP, and RhoHV to filter data. 
Needs csu_radartools, pyart, and astropy packages. 

Created Nov. 17, 2021

@author: mrocq
"""

#import the goods
import numpy as np
from csu_radartools import (csu_misc, csu_kdp)
import pyart
import os
from astropy.convolution import convolve, Gaussian2DKernel

#Adds a newly created field to the radar object.
def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID',
                              dz_field='ZC'):
    
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

#Extract data from radar file
def extract_unmasked_data(radar, field, bad=-32768):
    return radar.fields[field]['data'].filled(fill_value=bad)

#QC radar data
def radar_qc(file, sdp_thres=20, sdpz_thres = 10, despec_thres=4, ncp_thres=0.25, rho_thres=0.15):
       
    radar = pyart.io.read_cfradial(file, file_field_names=True)
    
    #extract reflectivity, differential phase from radar file
    dzN = extract_unmasked_data(radar, 'attenuation_corrected_reflectivity_h')
    dpN = extract_unmasked_data(radar, 'differential_phase')

    #calculate KDP, standard deviation of phase
    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(dp=dpN, dz=dzN, rng=rng2d/1000.0, thsd=12, gs=250.0, window=7, nfilter=2)

    #Apply standard deviation of phase masking, with dBZ threshold
    sdp_mask = csu_misc.differential_phase_filter(sdN, thresh_sdp=sdp_thres)
    sdp_mask = np.logical_and(sdp_mask, dzN <= sdpz_thres)
    
    bad = -32768
    dz_qc = 1.0 * dzN
    dz_qc[sdp_mask] = bad
    
    #Apply despeckling routine
    #mask_ds = csu_misc.despeckle(dz_qc, ngates=despec_thres)
    #dz_qc[mask_ds] = bad
    
    #Apply normalized coherent power masking after smoothing
    sm_g_rad = 5
    gauss_kernel = Gaussian2DKernel(sm_g_rad, sm_g_rad)
    ncp_smooth = convolve(radar.fields['normalized_coherent_power']['data'], gauss_kernel)
    ncp_mask = np.ma.getdata((ncp_smooth <= ncp_thres))
    total_mask = np.logical_or(sdp_mask, ncp_mask)
    dz_qc[total_mask] = bad

    #Apply correlation coefficient masking
    rho_mask = radar.fields['copol_correlation_coeff']['data'] <= rho_thres
    total_mask1 = np.logical_or(total_mask, rho_mask)
    dz_qc[total_mask1] = bad
        
    #Apply despeckling routine
    mask_ds = csu_misc.despeckle(dz_qc, ngates=despec_thres)
    #total_mask2 = np.logical_or(total_mask1, mask_ds)
    dz_qc[mask_ds] = bad
        
    #Add new QCed reflectivity field to radar file
    radar_qced = add_field_to_radar_object(dz_qc, radar, field_name='DZ_qc', units='dBZ', long_name='Reflectivity (QCed)',
                                   standard_name='DZ (QCed)', dz_field='attenuation_corrected_reflectivity_h')
    
    #Apply mask to other fields
    #VR
    dvN = extract_unmasked_data(radar, 'mean_doppler_velocity')
    vel_qc = 1.0 * dvN
    vel_qc[total_mask1] = bad
    
    radar_qced = add_field_to_radar_object(vel_qc, radar, field_name='VR_qc', units='m/s', long_name='Velocity (QCed)',
                                   standard_name='VR (QCed)', dz_field='attenuation_corrected_reflectivity_h')
    
    #ZDR
    drN = extract_unmasked_data(radar, 'differential_reflectivity')
    zdr_qc = 1.0 * drN
    zdr_qc[total_mask1] = bad
    
    radar_qced = add_field_to_radar_object(zdr_qc, radar, field_name='ZDR_qc', units='dB', long_name='Differential Reflectivity (QCed)',
                                   standard_name='ZDR (QCed)', dz_field='attenuation_corrected_reflectivity_h')
    
    #RhoHV
    drhN = extract_unmasked_data(radar, 'copol_correlation_coeff')
    rho_qc = 1.0 * drhN
    rho_qc[total_mask1] = bad
    
    radar_qced = add_field_to_radar_object(rho_qc, radar, field_name='RHOHV_qc', units='', long_name='Copol Correlation Ratio (QCed)',
                                   standard_name='RHOHV (QCed)', dz_field='attenuation_corrected_reflectivity_h')
    
    #PHIDP
    phi_qc = 1.0 * dpN
    phi_qc[total_mask1] = bad
    
    radar_qced = add_field_to_radar_object(phi_qc, radar, field_name='PHIDP_qc', units='deg', long_name='Differential Phase (QCed)',
                                   standard_name='PHIDP (QCed)', dz_field='attenuation_corrected_reflectivity_h')
    
    #KDP
    dkN = extract_unmasked_data(radar, 'specific_differential_phase')
    kdp_qc = 1.0 * dkN
    kdp_qc[total_mask1] = bad
    
    radar_qced = add_field_to_radar_object(kdp_qc, radar, field_name='KDP_qc', units='deg/km', long_name='Specific Differential Phase (QCed)',
                                   standard_name='KDP (QCed)', dz_field='attenuation_corrected_reflectivity_h')
    
    
    return (radar_qced)
