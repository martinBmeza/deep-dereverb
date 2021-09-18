import librosa
import acoustics
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import scipy
from sklearn.linear_model import LinearRegression
import glob

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list


def get_noise_level(cross_point, slope, env, DISTANCIA_AL_CRUCE):
    noise_init = cross_point + int(-DISTANCIA_AL_CRUCE/slope)

    if noise_init > len(env)*0.9:
        #tomar el 10%
        noise_floor = env[-int(len(env)*0.1):].mean()
        noise_floor_db = 10*np.log10(noise_floor)
    else:
        noise_floor = env[noise_init:].mean()
        noise_floor_db = 10*np.log10(noise_floor)
    return noise_floor_db

def linear_regression(n, arr):
    x = n.reshape(-1,1)
    y = arr
    model = LinearRegression().fit(x,y)
    intercept = model.intercept_
    slope = model.coef_
    return slope, intercept

def temporal_decompose(rir, fs, tau = 0.0025):
    t_d = np.argmax(rir) # direct path
    t_o = int((tau) * fs) #tolerance window in samples (2.5 ms)
    init_idx = t_d - t_o
    final_idx = t_d + t_o + 1

    if init_idx < 0:
        init_idx = 0
    if final_idx > len(rir)-1:
        final_idx = len(rir)-1

    early= rir[init_idx:final_idx]
    late = rir[final_idx:]
    delay = rir[:init_idx]
    return delay, early, late

def get_abs_envelope(arr, window_lenght = 50):
    arr = Series(arr).apply(lambda x:abs(x))
    arr_mean = arr.rolling(window=window_lenght,min_periods=1,center=True).mean()
    assert len(arr)==len(arr_mean)
    return arr_mean.to_numpy()

def normalize_rir(rir):
    """Normaliza un impulso, dividiendolo por su valor maximo, y quitando
    el delay inicial
    """
    #busco el valor maximo
    index_max = np.argmax(abs(rir))
    #Normalizo 
    rir = rir / rir[index_max]

    #Elimino el delay
    #rir = rir[index_max:]
    return rir


def estimated_fullband_decay(rir, fs):
    """Estima el decay-rate de un impulso entero, sin dividir
    en bandas. Se usa como parametro para la aumentacion
    por bandas."""

    delay, early, late = temporal_decompose(rir, fs)
    late_env = get_abs_envelope(late) # window lenght = 40

    #modelo de late field reverb
    t = np.linspace(0, len(late_env)/fs, len(late_env))
    popt, popv = scipy.optimize.curve_fit(curva_modelo, t, late_env, bounds=(0,1))
    return popt[1]


def sub_band_decompose(rir, fs, params):
    filterbank = generate_complementary_filterbank(**params)
    N = int(filterbank.shape[0]/2) #Tamaño del filtro
    rir_bands = np.empty((len(params['fc']), len(rir)))
    for band in range(len(params['fc'])):
        filtered_rir = scipy.signal.fftconvolve(rir, filterbank[:,band])[N:N+len(rir)]
        rir_bands[band,:] = filtered_rir
    return rir_bands


def generate_complementary_filterbank(
        fc=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
        fs=16000,
        filter_order=4,
        filter_length=16384,
        power=True):

    """Return a zero-phase power (or amplitude) complementary filterbank via Butterworth prototypes.
    Parameters:
        fc - filter center frequencies
        fs - sampling rate
        filter_order - order of the prototype Butterworth filters
        filter_length - length of the resulting zero-phase FIR filters
        power - boolean to set if the filter is power or amplitude complementary
    """
    fc = [i*np.power(2,1/2) for i in fc]
    # sort in increasing cutoff
    fc = np.sort(fc)

    assert fc[-1] <= fs/2

    numFilts = len(fc)
    nbins = filter_length
    signal_z1 = np.zeros(2 * nbins)
    signal_z1[0] = 1
    irBands = np.zeros((2 * nbins, numFilts))

    for i in range(numFilts - 1):
        wc = fc[i] / (fs/2.0)
        # if wc >= 1:
        #     wc = .999999

        B_low, A_low = scipy.signal.butter(filter_order, wc, btype='low')
        B_high, A_high = scipy.signal.butter(filter_order, wc, btype='high')


        # Store the low band
        irBands[:, i] = scipy.signal.lfilter(B_low, A_low, signal_z1)

        # Store the high
        signal_z1 = scipy.signal.lfilter(B_high, A_high, signal_z1)

        # Repeat for the last band of the filter bank
    irBands[:, -1] = signal_z1

    # Compute power complementary filters
    if power:
        ir2Bands = np.real(np.fft.ifft(np.square(np.abs(np.fft.fft(irBands, axis=0))), axis=0))
    else:
        ir2Bands = np.real(np.fft.ifft(np.abs(np.abs(np.fft.fft(irBands, axis=0))), axis=0))

    ir2Bands = np.concatenate((ir2Bands[nbins:(2 * nbins), :], ir2Bands[0:nbins, :]), axis=0)

    return ir2Bands

def plot_bankfilter(params):
    bankfilter = generate_complementary_filterbank(**params)
    h_sum = np.zeros(8000)
    fs=16000
    for banda in range(7): 
        w, h = scipy.signal.freqz(bankfilter[:,banda], worN=8000)
        h_sum += abs(h)
        plt.semilogx((w/np.pi)*(fs/2), abs(h), linewidth=2, label = str(params['fc'][banda])+' Hz')

    plt.semilogx((w/np.pi)*(fs/2), h_sum)
    plt.xticks([125, 250, 500, 1000, 2000, 4000, 8000], ['125', '250', '500', '1000', '2000', '4000', '8000'])
    plt.legend(prop={'size': 24})
    plt.grid()
    plt.subplots_adjust(right=1.5, top=1.5)
    plt.show()

def get_envelope(arr, window_lenght):
    #VER DE CAMBIAR MAX POR MEAN Y MEDIAN
    #arr = Series(arr).apply(lambda x:x**2)
    arr = Series(arr).apply(lambda x:abs(x))

    arr_mean = arr.rolling(window=window_lenght,min_periods=1,center=True).max()
    assert len(arr)==len(arr_mean)
    return arr_mean.to_numpy()

def get_valid_interval(arr, init_value, final_value):
    samples = np.arange(len(arr))
    onset = np.where(arr < init_value)[0][0]
    arr_aux = arr[onset:]

    final = np.where(arr_aux < final_value)[0][0]
    arr_aux2 = arr_aux[:final]

    offset = len(arr) - len(arr_aux)
    samples = samples[onset:(final+offset)]
    return arr_aux2, samples

def Lundeby_method(rir, fs):
    #print('Banda: '+str(banda))
    #constantes
    EPS = np.finfo(float).eps
    TIME_INTERVAL = 160 #10 ms
    DISTANCIA_AL_PISO = 5
    N_INTERVALOS_10DB = 10
    DISTANCIA_AL_CRUCE = 5
    RANGO_DINAMICO = 10
    
    #Delay direct
    max_idx = np.argmax(abs(rir))
    
    if max_idx == 0:
        ADD_INIT = False
        rir = rir
    else:
        ADD_INIT = True
        delay = rir[:max_idx]
        rir = rir[max_idx:]

    #Envolvente de la señal al cuadrado
    env = get_envelope(rir, TIME_INTERVAL)
    rir_squared = rir**2
    rir_db = 10*np.log10(rir_squared+EPS)
    env_db = 10*np.log10(env+EPS)
    n = np.arange(len(env_db))

    #Primera estimacion usando el 10%
    noise_floor = env[-int(len(env)*0.1):].mean()
    noise_floor_db = 10*np.log10(noise_floor)

    init_value = env_db.max()
    final_value = noise_floor_db + DISTANCIA_AL_PISO
    
    env_db_chunk, n_chunk = get_valid_interval(env_db, init_value, final_value)
    
    slope, intercept = linear_regression(n_chunk, env_db_chunk)

    cross_point = int((noise_floor_db - intercept)/slope)

    time_interval = int((-10/slope)/N_INTERVALOS_10DB)

    env = get_envelope(rir, time_interval)

    #Segmento para calcular el nuevo piso de ruido
    iteracion = 0
    delta_level = 1.0

    while (iteracion <6 and delta_level > 0.2):
        
        #limito el punto de cruce al largo del audio
        if cross_point > len(env_db)-1:
            cross_level_1 = env_db[-1]
        else:
            cross_level_1 = env_db[cross_point]

        noise_floor_db = get_noise_level(cross_point, slope, env, DISTANCIA_AL_CRUCE)

        init_value = noise_floor_db + DISTANCIA_AL_PISO
        if init_value - RANGO_DINAMICO < np.min(env_db):
            final_value = env_db[-2]
        else:
            final_value = init_value - RANGO_DINAMICO
        env_db_chunk, n_chunk = get_valid_interval(env_db, init_value, final_value)
        slope, intercept = linear_regression(n_chunk, env_db_chunk)

        cross_point = int((noise_floor_db - intercept)/slope)
        
        if cross_point > len(env_db)-1:
            cross_level_2 = env_db[-1]
        else:
            cross_level_2 = env_db[cross_point]

        delta_level = abs(cross_level_1 - cross_level_2)
        iteracion += 1
        
    rir_cut = rir[:cross_point]
    
    if ADD_INIT:
        rir_salida = np.concatenate((delay, rir_cut))
        rir_plot = np.concatenate((delay, rir))
    else:
        rir_salida = rir_cut
        rir_plot = rir
        
    cross_point_compensado = len(rir_salida)
    #axs[banda].plot(librosa.amplitude_to_db(rir_plot), label='Entrada')
    #axs[banda].plot(librosa.amplitude_to_db(rir_salida), label='Punto de corte')
    #axs[banda].legend()
    #axs[banda].set_title('BANDA '+str(banda), fontsize=18)
    return cross_point_compensado


def curva_modelo(t, Am, decay_rate, noise_floor):
    ones = np.ones(len(t))
    modelo = Am * np.exp(-t/decay_rate) * ones + (noise_floor*ones)
    return modelo

def get_abs_max_envelope(arr, window_lenght = 500):
    arr = Series(arr).apply(lambda x:abs(x))
    arr_mean = arr.rolling(window=window_lenght,min_periods=1,center=True).median()
    assert len(arr)==len(arr_mean)
    return arr_mean.to_numpy()

def estim_params(late, cross_point, fs):
    
    late_env = get_abs_max_envelope(late) # window lenght = 40
    late_env_valid = late_env[:cross_point]
    
    #ESTOY HACIENDO LA ESTIMACION SOLO CON LO QUE NO TIENE RUIDO.(LATE ENV VALID)
    t = np.linspace(0, len(late_env_valid)/fs, len(late_env_valid))
    t_entera = np.linspace(0, len(late_env)/fs, len(late_env))
    popt, popv = scipy.optimize.curve_fit(curva_modelo, t, late_env_valid, bounds=(0,1))
    
    estim_params = {'Am':popt[0], 'decay_rate':popt[1], 'noise_floor':popt[2]} 
    #axs[banda].plot(librosa.amplitude_to_db(late_env_valid), label='Envolvente a estimar')
    #axs[banda].plot(librosa.amplitude_to_db(curva_modelo(t_entera, **estim_params)), linewidth=3, label='Estimacion parametrica')
    return estim_params

def curva_noiseless(t, Am, decay_rate):
    noise = np.random.normal(0,1,len(t))
    modelo = Am * np.exp(-t/decay_rate) * noise
    return modelo

def cross_fade(señal_1, señal_2, fs, cross_point):
    """
    señal 1 se atenua luego del cross point
    señal 2 se amplifica luego del cross point
    """
    #print(str(len(señal_1)) +'  '+str(len(señal_2))+'  '+str(cross_point))
    largo = int(50 * 0.001 * 16000) # 800 muestras
    if 2*largo > len(señal_1)-cross_point:
        return señal_1
    ventana = scipy.signal.hann(largo)
    fade_in, fade_out = ventana[:int(largo/2)], ventana[int(largo/2):]
    
    ventana_atenuante = np.concatenate((np.ones(cross_point-int(fade_out.size/2)),
                                        fade_out,
                                        np.zeros(len(señal_1)-cross_point-int(fade_out.size/2))))

    ventana_amplificadora = np.concatenate((np.zeros(cross_point-int(fade_out.size/2)),
                                            fade_in, 
                                            np.ones(len(señal_2)-cross_point-int(fade_out.size/2))))
    return (señal_1*ventana_atenuante) + (señal_2*ventana_amplificadora)

def noise_crossfade(rir, estim_params, cross_point, fs):
    t = np.linspace(0, len(rir)/fs, len(rir))
    rir_noiseless = curva_noiseless(t, estim_params['Am'], estim_params['decay_rate'])
    rir_denoised = cross_fade(rir, rir_noiseless, fs, cross_point)

    #axs[banda].plot(t,librosa.amplitude_to_db(rir_bands[banda,:]), linewidth=3, label='Estimacion parametrica')
    #axs[banda].plot(t,librosa.amplitude_to_db(rir_noiseless), label='Envolvente sin ruido')
    #axs[banda].plot(t,librosa.amplitude_to_db(curva_modelo(t, **estim_params)), linewidth=3, label='Estimacion parametrica')
    #axs[banda].plot(t,librosa.amplitude_to_db(rir_denoised), linewidth=3, label='Crossfadeado')
    #axs[banda].legend()
    return rir_denoised


def augmentation(rir, estim_params, estim_fullband_decay ,TR60_desired, fs):

    t = np.linspace(0, len(rir)/fs, len(rir))
    decay_rate_d = TR60_desired / (np.log(1000))
    ratio = decay_rate_d / estim_fullband_decay
    t_md = ratio * estim_params['decay_rate']

    #Augmentation
    rir_aug = rir * np.exp(-t*((estim_params['decay_rate']-t_md)/(estim_params['decay_rate']*t_md)))

    #axs.plot(t,librosa.amplitude_to_db(rir), linewidth=3, label='Original')
    #axs.plot(t,librosa.amplitude_to_db(rir_aug), linewidth=3, label='Aug')
    return rir_aug

def tr_augmentation(rir_entrada, fs, TR_DESEADO):
    params = {'fc' : [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
              'fs' : 16000,
              'filter_order' : 4,
              'filter_length' : 16384,
              'power' : True}

    rir_entrada = normalize_rir(rir_entrada)
    delay, early, rir = temporal_decompose(rir_entrada,fs)

    estim_fullband_decay = estimated_fullband_decay(rir, fs)

    rir_bands = sub_band_decompose(rir, fs, params)
    rir_band_augs = np.empty(rir_bands.shape)
    for banda in range(rir_bands.shape[0]):
        cross_point = Lundeby_method(rir_bands[banda,:], fs)
        parameters = estim_params(rir_bands[banda,:], cross_point, fs)
        rir_band_denoised = noise_crossfade(rir_bands[banda,:],
                                            parameters,
                                            cross_point,
                                            fs)
        rir_band_augs[banda,:] = augmentation(rir_band_denoised,
                                    parameters,
                                    estim_fullband_decay,
                                    TR_DESEADO,
                                    fs)
    rir_aug = np.sum(rir_band_augs, axis=0)
    rir_aug = np.concatenate((delay, early, rir_aug))
    return rir_aug, fs
