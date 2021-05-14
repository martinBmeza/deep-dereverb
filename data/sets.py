
'''Formar los respectivos conjuntos de train, validation y test sets para cada experimento'''
import tqdm
import glob
import random
import pickle

data_folder = '/mnt/datasets/'


def get_list(path, rand_select = 0):
    
    data_types = ('.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3') 
    search_path = path + '/**/*'
    audio_list = []

    print('Scanning for audio files...')
    for types in tqdm.tqdm(data_types): 
        audio_list.extend(glob.glob(search_path+types, recursive = True))
    
    if rand_select !=0:
        audio_list = random.sample(audio_list, rand_select)

    return audio_list


"""
EXPERIMENTO 1: Comparacion con el estado del arte
    
    Training_set:
        clean_speech: TIMIT, 4620 utts from train set
        sim_RIR: 15 x Room1 x {500, 750, 1000 ms}, total 45 RIRs
        real_RIR: 50 GreatHall, 50 Ocatagon, 40 Classroom, total 140 RIRs

    Val_set: 

    Test_set:
        clean_speech: TIMIT, 192 utts from test set
        sim_RIR: 10 x Room1 x {500, 750, 1000}, 10 x Room2 x {500, 750, 1000}
        real_RIR: 10 from each room

Room1: 8 x 6 x 4 mts 
Room2: 6 x 4 x 3.2 mts
"""
exp1_sets = {}

#CLEAN SPEECH
clean_speech_path =  data_folder + 'clean_voice/TIMIT/'
clean_train_list = get_list(clean_speech_path+'TRAIN')
clean_test_list = get_list(clean_speech_path+'TEST', rand_select = 192)
exp1_sets['clean_train'] = clean_train_list
exp1_sets['clean_test'] = clean_test_list


#RIRs REALES
real_rir_path = data_folder + 'impulsos/reales/'
real_greathall = get_list(real_rir_path+'greathallOmni', rand_select = 60)
real_octagon = get_list(real_rir_path+'octagonOmni', rand_select = 60)
real_classroom = get_list(real_rir_path+'classroomOmni', rand_select = 50)

real_greathall_test = random.sample(real_greathall, 10)
real_octagon_test = random.sample(real_octagon, 10)
real_classroom_test = random.sample(real_classroom, 10)

real_greathall_train = [i for i in real_greathall if i not in real_greathall_test]
real_octagon_train = [i for i in real_octagon if i not in real_octagon_test]
real_classroom_train = [i for i in real_classroom if i not in real_classroom_test]
real_train = real_greathall_train + real_octagon_train + real_classroom_train


real_test = {'greathall': real_greathall_test,
             'octagon' : real_octagon_test,
             'classroom': real_classroom_test}

exp1_sets['real_train'] = real_train
exp1_sets['real_test'] = real_test

#RIRs SINTETICAS
sim_rir_path = data_folder + 'impulsos/sinteticos/'
sim_list_r1_500 = get_list(sim_rir_path+'room1/500')
sim_list_r1_750 = get_list(sim_rir_path+'room1/750')
sim_list_r1_1000 = get_list(sim_rir_path+'room1/1000')

sim_train_list = random.sample(sim_list_r1_500, 15) +  random.sample(sim_list_r1_500, 15) + random.sample(sim_list_r1_500, 15)

sim_test_r2_500 = get_list(sim_rir_path+'room2/500')
sim_test_r2_750 = get_list(sim_rir_path+'room2/750')
sim_test_r2_1000 = get_list(sim_rir_path+'room2/1000')
room2 = {'500' : sim_test_r2_500,
         '750' : sim_test_r2_750,
         '1000' : sim_test_r2_1000}

sim_test_r1_500 = [i for i in sim_list_r1_500 if i not in sim_train_list] 
sim_test_r1_750 = [i for i in sim_list_r1_750 if i not in sim_train_list] 
sim_test_r1_1000 = [i for i in sim_list_r1_1000 if i not in sim_train_list] 
room1 = {'500' : sim_test_r1_500,
         '750' : sim_test_r1_750,
         '1000' : sim_test_r1_1000}

sim_test = {'room1' : room1, 'room2' : room2}
exp1_sets['sim_train'] = sim_train_list 
exp1_sets['sim_test'] = sim_test

out_path = data_folder + 'npy_data/experimento_1/'
exp1_sets['out_path'] = out_path 

with open('exp1.pkl', 'wb') as f:
    pickle.dump(exp1_sets, f)


