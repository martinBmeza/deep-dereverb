import librosa
import soundfile as sf
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from scipy.signal import fftconvolve

def reverb_multiprocessing(audio_list, rir_list, save):
    """
    Genera audios con reverberacion partiendo de una lista de audios
    anecoicos y una lista de respuesta al impulso disponibles. Luego 
    de generarlos, los almacena en un directorio.
    
    Parametros:
    ----------
    audio_list : list
        Lista en donde cada elemento es una lista con dos
        items. El primero es el path del audio de habla, y el segundo es la numeracion
        correspondiente para cada segmento a guardar.
    rir_list : list
        Lista que contiene los paths de las respuestas al impulso disponibles para 
        generar la reverberacion.
    save : str
        Path en donde guardar los segmentos de audio reverberado generados.
    """

    tasks = [[audio_list[i], np.random.choice(rir_list), save] for i in range(len(audio_list))]
    pool = mp.Pool(processes=5)
    for _ in tqdm(pool.imap(reverberacion, tasks), total=len(tasks)):
        pass
    pool.close()
    pool.join()

def clean_multiprocessing(audio_list, save):
    """
    Lee archivos de audio de habla y los segmenta. Luego, 
    los almacena en un directorio.
    
    Parametros:
    ----------
    audio_list : list
        Lista en donde cada elemento es una lista con dos
        items. El primero es el path del audio de habla, y el segundo es la numeracion
        correspondiente para cada segmento a guardar.
    save : str
        Path en donde guardar los segmentos de audio generados.
    """

    tasks = [[audio_list[i], save] for i in range(len(audio_list))]
    pool = mp.Pool(processes=5)
    for _ in tqdm(pool.imap(clean_cut, tasks), total=len(tasks)):
        pass
    pool.close()
    pool.join()

def clean_cut(task):
    """
    Dado un audio de habla, realiza la lectura, segmentacion y
    guardado del audio en un directorio determinado.

    Parametros:
    ----------
    task : list
        Lista que contiene los argumentos necesarios para realizar
        el procesamiento. EL primer elemento es una lista que contiene
        el path del audio y la numeracion determinada. El segundo
        item, contiene el path de guardado de los audios generados.
    """
    # Desestructuracion
    speech_path, numbers = task[0][0], task[0][1]
    clean_save = task[1]
    
    # Cargo el audio
    clean, speech_fs = librosa.load(speech_path, sr=16000)

    ventana = int(32640)
    # Cortado y guardado
    for i in range(len(clean)//ventana):
        clean_cut = clean[int(ventana*i):int(ventana*(i+1))]
        clean_fn = clean_save+'{:06d}'.format(numbers[i])+ '.npy'
        np.save(clean_fn, clean_cut)


def reverberacion(task):
    """
    Dado un audio de habla y una respuesta al impulso,
    realiza la lectura, reverberacion, segmentacion y
    guardado del audio en un directorio determinado.

    Parametros:
    ----------
    task : list
        Lista que contiene los argumentos necesarios para realizar
        el procesamiento. EL primer elemento es una lista que contiene
        el path del audio y la numeracion determinada. El segundo
        item corresponde al path de la respuesta al impulso requerida, y
        el tercer item contiene el path de guardado de los audios generados.
    """

    # Desestructuracion
    speech_path, numbers = task[0][0], task[0][1]
    rir_path = task[1]
    reverb_save = task[2]
    
    # Cargo los audios
    speech, speech_fs = librosa.load(speech_path, sr=16000)
    rir, rir_fs = librosa.load(rir_path, sr=16000)

    # Reverberacion
    speech = speech / np.max(abs(speech))
    clean, reverb = generar_reverb(speech, rir)
    #clean = speech
    ventana = int(32640)
    
    # Cortado y guardado
    for i in range(len(clean)//ventana):
        reverb_cut = reverb[int(ventana*i):int(ventana*(i+1))]
        reverb_fn = reverb_save+'{:06d}'.format(numbers[i])+ '.npy'
        np.save(reverb_fn, reverb_cut)


def generar_reverb(speech, rir):
    """
    Realiza la reverberacion entre una señal de habla y una 
    respuesta al impulso. Se asume que ambos arrays comparten
    la misma frecuencia de muestreo, y que la informacion de 
    amplitud puede ser normalizada.

    Parametros:
    -----------
    speech : array
        Numpy array correspondiente a la lectura de del audio
        de habla.
    rir : array
        Numpy array correspondiente a la lectura de la respuesta
        al impulso.
    
    Salida:
    -------
    reverb : array
        Numpy array correspondiente al audio reverberado.
    """

    # Normalizo el impulso
    rir = rir / np.max(abs(rir))
    rir = rir[np.argmax(abs(rir)):]  
    
    # Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir)[:len(speech)]
    #early_reverb = fftconvolve(speech, rir[:320])[:len(speech)]
    early_reverb = speech
    
    return early_reverb, reverb


def audio_number(audio_list, winsize=32640):
    """
    Dada una lista de audios, conforma una nueva lista en donde
    cada elemento de esta se componga del path del audio, junto 
    con un array de numeros indicando la numeracion de cada 
    segmento que puede ser generado para cumplir con un determinado
    tamaño de ventana. 

    Parametros:
    -----------
    audio_list : list
        Lista de paths correspondiente a los archivos de audio de habla.
    winsize : int 
        Tamaño de la ventana temporal que se busca en cada segmento de audio
        generado. Por defecto es de 32640 muestras, para obtener una stft con
        256 unidades temporales correspondiente a una ventana de 512 muestras
        con un tamaño de salto de 128 muestras. 
    
    Salida : lista
        Lista donde cada elemento se conforma por un path de audio y un array 
        que contiene la numeracion correspondiente para cada segmento de audio
        generado.
    -------
    """

    audio_number = [] 
    n = 0
    errores = []
    for audio_path in tqdm(audio_list):
        nframes = sf.info(audio_path, verbose=True).frames
        nchunks = nframes // winsize
        if nchunks == 0:
            continue 
        num = np.arange(n, n + nchunks)
        audio_number.append([audio_path, num])
        n += nchunks 
    return audio_number   





