import os
import librosa
import pandas
import math
import json

diagnosis = pandas.read_csv('E:\Jatin\Projects\LBS Project\ICBHI_final_database\patient_diagnosis.csv')
diagnosis = diagnosis.to_dict()
ids=list(diagnosis['ID'].values())
print(ids)
symp=list(diagnosis['Symptom'].values())
print(symp)
json_path = "E:/Jatin/Projects/LBS Project/Test/venv/data.json"
dataset_path = "E:\Jatin\Projects\LBS Project\ICBHI_final_database\ICBHI_final_database"

def slicing_track(duration, s, num_segment=5, hop_length=512):
    samples_per_track = 22050 * duration
    num_samples_per_segment = int(samples_per_track/num_segment)
    start_sample = num_samples_per_segment * s
    final_sample = start_sample + num_samples_per_segment
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)
    return start_sample, final_sample, expected_num_mfcc_vectors_per_segment




def save_mfccs(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segment=5):

    #Making dictionary to store data
    data={
        "mapping":[],
        "mfccs":[],
        "labels":[]
    }

    #loading audio file

    for i,(dirpath,dirname,filenames) in enumerate(os.walk(dataset_path)):
        print(dirpath, filenames)
        for filename in filenames:
            #print(filename[-4:])
            if filename[-4:] == '.wav':
                #print(filename[-4:])
                filepath=os.path.join(dirpath, filename)
                print(filename[:3])
                #print(int(filename[:3]) in ids)
                if (int(filename[:3]) in ids):
                    index=ids.index(int(filename[:3]))
                    print(index)
                    symptom = symp[index]
                    print(symptom)
                    if symptom not in data['mapping']:
                        data['mapping'].append(symptom)

                signal,sr=librosa.load(filepath, sr=22050)
                duration = librosa.get_duration(y=signal, sr=sr)

                #process segments extracting mfcc and storing data
                for s in range(num_segment):
                    start_sample, final_sample, expected_mfcc = slicing_track(duration=duration, s=s, num_segment=5, hop_length=hop_length)

                    mfcc = librosa.feature.mfcc(signal[start_sample:final_sample], sr=sr,n_fft=n_fft,
                                            n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc=mfcc.T

                    if len(mfcc)==expected_mfcc:
                        data['mfccs'].append(mfcc.tolist())
                        index_symptom = data['mapping'].index(symptom)
                        data['labels'].append(index_symptom)


    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        print(len(data["labels"],data["mfccs"]))



if __name__ == "__main__":
    save_mfccs(dataset_path=dataset_path, json_path=json_path)
