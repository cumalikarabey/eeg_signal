# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 14:51:08 2021

@author: casper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle    
from scipy import signal 
from scipy.signal import find_peaks
from scipy import stats

def sinyal_isle(participant_id):
    peaks = []
    alpha_data = []
    beta_data = []
    delta_data = []
    gamma_data = []
    theta_data = []
    alpha_data_median = []
    beta_data_median = []
    delta_data_median = []
    gamma_data_median = []
    theta_data_median = []
    alpha_res = []
    beta_res = []
    delta_res = []
    gamma_res = []
    theta_res = []
    etiketler_data = []
    kullanici_degerlendirme = []
    video_listesi = []
    if participant_id < 1 or participant_id > 32:
        return print("1-32 arasında bir ID giriniz.")
        
    '''Data'''
    online_ratings = pd.read_csv('online_ratings.csv')
    participant_questionnaire =  pd.read_csv('participant_questionnaire.csv')
    participant_ratings = pd.read_csv('participant_ratings.csv')
    video_list = pd.read_csv('video_list.csv')
    with open('data/s'+str(participant_id)+'.dat', 'rb') as f:
        kisi = pickle.load(f, encoding='latin1')
    
        
    '''Signal'''
    x_label = np.arange(0,8064).reshape(1,8064)
    y = kisi["data"]
    
    
    
    channels_alpha = []     #alpha kanalı
    channels_beta = []     #beta kanalı
    channels_delta = []     #delta kanalı
    channels_gamma = []     #gamma kanalı
    channels_theta = []     #theta kanalı
    
    
    average_psd_alpha = []  #kanlların ortalama güç yoğunlukları
    average_psd_beta = []
    average_psd_delta = []
    average_psd_gamma = []
    average_psd_theta = []
    
    
    peaks_data = []         #keskin uç noktaları
    
    
    channels_alpha_new = [] 
    channels_beta_new = []
    channels_delta_new = []
    channels_gamma_new = []
    channels_theta_new = []
    
    
    var_all = []
    var_alpha = [] 
    var_beta = []
    var_delta = []
    var_gamma = []
    var_theta = []
    
    
    sts_signal = []
    alpha_sts = []
    beta_sts = []
    delta_sts = []
    gamma_sts = []
    theta_sts = []
    
    
    
    mean_all =[]
    psd_all = []
    
    
    alpha_median = []
    beta_median = []
    delta_median = []
    gamma_median = []
    theta_median = []
    all_median = []
    
    
    
    for z in np.arange(0,40,1):
        
    
        for i in np.arange(0,32,1):
            
            y_label = y[z, i:i+1, :8064]
            
            '''Filter Design'''
            
            '''Alpha'''
            sos_alpha = signal.butter(7, [8,12], 'bandpass' , fs = 128, output='sos')
            filtered_alpha = signal.sosfilt(sos_alpha, y_label[0])
            
            '''Beta'''
            sos_beta = signal.butter(7, [12,30], 'bandpass' , fs = 128, output='sos')
            filtered_beta = signal.sosfilt(sos_beta, y_label[0])
            
            '''Delta'''
            sos_delta = signal.butter(7, [0.5,4], 'bandpass' , fs = 128, output='sos')
            filtered_delta = signal.sosfilt(sos_delta, y_label[0])
            
            '''Gamma'''
            sos_gamma = signal.butter(7, [30,45], 'bandpass' , fs = 128, output='sos')
            filtered_gamma = signal.sosfilt(sos_gamma, y_label[0])
            
            '''Theta'''
            sos_theta = signal.butter(7, [4,8], 'bandpass' , fs = 128, output='sos')
            filtered_theta = signal.sosfilt(sos_theta, y_label[0])
            
            
            
            '''Average Power Spectral Density'''
            
            '''Alpha'''
            f, PSD_alpha = signal.periodogram(filtered_alpha, 128)
            apsd_alpha = np.mean(PSD_alpha)
            
            
            '''Beta'''
            f, PSD_beta = signal.periodogram(filtered_beta, 128)
            apsd_beta = np.mean(PSD_beta)
            
            '''Delta'''
            f, PSD_delta = signal.periodogram(filtered_delta, 128)
            apsd_delta = np.mean(PSD_delta)
            
            '''Gamma'''
            f, PSD_gamma = signal.periodogram(filtered_gamma, 128)
            apsd_gamma = np.mean(PSD_gamma)
            
            '''Theta'''
            f, PSD_theta = signal.periodogram(filtered_theta, 128)
            apsd_theta = np.mean(PSD_theta)
            
            
            
            '''Resample Signals'''
            
            f_alpha = signal.resample(filtered_alpha, 1000)
            f_beta = signal.resample(filtered_beta, 1000)
            f_delta = signal.resample(filtered_delta, 1000)
            f_gamma = signal.resample(filtered_gamma, 1000)
            f_theta = signal.resample(filtered_theta, 1000)
            
            
            '''Spikes'''
            
            peaks, _ = find_peaks(y_label[0],threshold= np.mean(y_label[0])+3 ,distance=7)
            peaks_data.append(peaks)
            a = np.diff(peaks)     #peak'ler arası mesafe
            
            
            '''Variance'''
            alpha_varyans = stats.tvar(filtered_alpha)
            beta_varyans =  stats.tvar(filtered_beta)
            delta_varyans = stats.tvar(filtered_delta)
            gamma_varyans = stats.tvar(filtered_gamma)
            theta_varyans = stats.tvar(filtered_theta)
            
            all_varyans = stats.tvar(y_label[0])
           
            
            '''Standard Deviation'''
            stan_dev_alpha = stats.tstd(filtered_alpha)
            stan_dev_beta = stats.tstd(filtered_beta)
            stan_dev_delta = stats.tstd(filtered_delta)
            stan_dev_gamma = stats.tstd(filtered_gamma)
            stan_dev_theta = stats.tstd(filtered_theta)
            
            stan_dev = stats.tstd(y_label[0])
            
            f, PSD_all = signal.periodogram(y_label[0], 128)
            apsd_all = np.mean(PSD_all)
            psd_all.append(PSD_all)
            mean_all.append(apsd_all)
            
            alpha_med = signal.medfilt(filtered_alpha,kernel_size=33)
            beta_med = signal.medfilt(filtered_beta,kernel_size=33)
            delta_med = signal.medfilt(filtered_delta,kernel_size=33)
            gamma_med = signal.medfilt(filtered_gamma,kernel_size=33)
            theta_med = signal.medfilt(filtered_theta,kernel_size=33)

            all_med = signal.medfilt(y_label[0],kernel_size=129)
            all_median.append(all_med)
            alpha_median.append(alpha_med)
            beta_median.append(beta_med)
            delta_median.append(delta_med)
            gamma_median.append(gamma_med)
            theta_median.append(theta_med)
           
            
            '''Data Append'''
            
            channels_alpha_new.append(f_alpha)
            channels_beta_new.append(f_beta)
            channels_delta_new.append(f_delta)
            channels_gamma_new.append(f_gamma)
            channels_theta_new.append(f_theta)
            
            channels_alpha.append(filtered_alpha)
            channels_beta.append(filtered_beta)
            channels_delta.append(filtered_delta)
            channels_gamma.append(filtered_gamma)
            channels_theta.append(filtered_theta)
            
            average_psd_alpha.append(apsd_alpha)
            average_psd_beta.append(apsd_beta)
            average_psd_delta.append(apsd_delta)
            average_psd_gamma.append(apsd_gamma)
            average_psd_theta.append(apsd_theta)
            
            var_alpha.append(alpha_varyans)
            var_beta.append(beta_varyans)
            var_delta.append(delta_varyans)
            var_gamma.append(gamma_varyans)
            var_theta.append(theta_varyans)
            
            var_all.append(all_varyans)
            
            alpha_sts.append(stan_dev_alpha)
            beta_sts.append(stan_dev_beta)
            delta_sts.append(stan_dev_delta)
            gamma_sts.append(stan_dev_gamma)
            theta_sts.append(stan_dev_theta)
            
            sts_signal.append(stan_dev)
    
    peaks_all = []
    for i in np.arange(0,len(peaks_data)):
        peak_len = len(peaks_data[i])
        peaks_all.append(peak_len)
    
    peaks_all = pd.DataFrame(peaks_all)
    sts_signal = pd.DataFrame(sts_signal)
    var_all = pd.DataFrame(var_all)
    mean_all = pd.DataFrame(mean_all)
    psd_all = pd.DataFrame(psd_all)
    all_median = pd.DataFrame(all_median)
    peaks_sts = pd.concat([peaks_all, sts_signal, var_all, mean_all],axis=1)
    
    
    video_id = video_list.iloc[:120, 1:2].values
    avg_arousal = video_list.iloc[:120, 15:16].values
    avg_valence = video_list.iloc[:120, 10:11].values
    arousal = []
    valence = []
    for i in np.arange(0,len(video_id)):
        if video_id[i]>0:
            arousal.append(avg_arousal[i])
            valence.append(avg_valence[i])
        
    
    duygular = []
    for i in np.arange(0, len(arousal)):
        if valence[i]>5:
            if arousal[i]>5:
                duygular.append("HAHV")
            else:
                duygular.append("LAHV")
        else:
            if arousal[i]<5:
                duygular.append("LALV")
            else:
                duygular.append("HALV")        
    
    video_num = video_id[~np.isnan(video_id)]
    duygular = pd.DataFrame(duygular, columns=['Response'])
    arousal = pd.DataFrame(arousal, columns=['AVG_arousal'])
    valence = pd.DataFrame(valence, columns=['AVG_valence'])
    video_num = pd.DataFrame(video_num, columns=['experiment_id'])
    video_deger_etiket = pd.concat([video_num,arousal,valence,duygular], axis=1)
    

    
    etiketler = []
    for i in np.arange(0, 40):
        for z in np.arange(0,len(video_deger_etiket)):
            if participant_ratings["Experiment_id"][(participant_id-1)*40 + i] == video_deger_etiket["experiment_id"][z]: #5. kisi icin video siralarini almak icin 160+i alindi
                for z2 in np.arange(0,32):
                    etiketler.append(video_deger_etiket["Response"][z])
    
    
    
    
    alpha = pd.DataFrame(data=channels_alpha)
    beta = pd.DataFrame(data=channels_beta)
    delta = pd.DataFrame(data=channels_delta)
    gamma = pd.DataFrame(data=channels_gamma)
    theta = pd.DataFrame(data=channels_theta)
    
    var_alpha = pd.DataFrame(var_alpha, columns=['Alpha VAR'])
    var_beta = pd.DataFrame(var_beta, columns=['Beta VAR'])
    var_delta = pd.DataFrame(var_delta, columns=['Delta VAR'])
    var_gamma = pd.DataFrame(var_gamma, columns=['Gamma VAR'])
    var_theta = pd.DataFrame(var_theta, columns=['Theta VAR'])
    var_kanal = pd.concat([var_alpha, var_beta, var_delta, var_gamma, var_theta], axis=1)
    
    average_psd_alpha = pd.DataFrame(data=average_psd_alpha, columns=['Alpha MEAN'])
    average_psd_beta = pd.DataFrame(data=average_psd_beta, columns=['Beta MEAN'])
    average_psd_delta = pd.DataFrame(data=average_psd_delta, columns=['Delta MEAN'])
    average_psd_gamma = pd.DataFrame(data=average_psd_gamma, columns=['Gamma MEAN'])
    average_psd_theta = pd.DataFrame(data=average_psd_theta , columns=['Theta MEAN'])
    sts_signal = pd.DataFrame(sts_signal, columns=['Stan_Deviation'])
    etiketler = pd.DataFrame(etiketler, columns=["Response"])
    av_psd_kanal = pd.concat([average_psd_alpha, average_psd_beta, average_psd_delta, average_psd_gamma, average_psd_theta], axis=1)
    var_mean = pd.concat([var_kanal, av_psd_kanal], axis=1)
    peaks_all = pd.DataFrame(peaks_all)
    
    
    average_psd_alpha = pd.DataFrame(data=average_psd_alpha)
    average_psd_beta = pd.DataFrame(data=average_psd_beta)
    average_psd_delta = pd.DataFrame(data=average_psd_delta)
    average_psd_gamma = pd.DataFrame(data=average_psd_gamma)
    average_psd_theta = pd.DataFrame(data=average_psd_theta)
    alpha_sts = pd.DataFrame(alpha_sts)
    beta_sts = pd.DataFrame(beta_sts)
    delta_sts = pd.DataFrame(delta_sts)
    gamma_sts = pd.DataFrame(gamma_sts)
    theta_sts = pd.DataFrame(theta_sts)

    peaks_data = pd.DataFrame(data=peaks_data)
    
    alpha_median = pd.DataFrame(alpha_median)
    beta_median = pd.DataFrame(beta_median)
    delta_median = pd.DataFrame(delta_median)
    gamma_median = pd.DataFrame(gamma_median)
    theta_median = pd.DataFrame(theta_median)

    channels_alpha_new = pd.DataFrame(channels_alpha_new)
    channels_beta_new = pd.DataFrame(channels_beta_new)
    channels_delta_new = pd.DataFrame(channels_delta_new)
    channels_gamma_new = pd.DataFrame(channels_gamma_new)
    channels_theta_new = pd.DataFrame(channels_theta_new)

    alpha = pd.concat([alpha, var_alpha, average_psd_alpha, alpha_sts], axis=1)
    beta = pd.concat([beta, var_beta, average_psd_beta, beta_sts], axis=1)
    delta = pd.concat([delta, var_delta, average_psd_delta, delta_sts], axis=1)
    gamma = pd.concat([gamma, var_gamma, average_psd_gamma, gamma_sts], axis=1)
    theta = pd.concat([theta, var_theta, average_psd_theta, theta_sts], axis=1)
    
    alpha_m = pd.concat([alpha_median, var_alpha, average_psd_alpha, alpha_sts], axis=1)
    beta_m = pd.concat([beta_median, var_beta, average_psd_beta, beta_sts], axis=1)
    delta_m = pd.concat([delta_median, var_delta, average_psd_delta, delta_sts], axis=1)
    gamma_m = pd.concat([gamma_median, var_gamma, average_psd_gamma, gamma_sts], axis=1)
    theta_m = pd.concat([theta_median, var_theta, average_psd_theta, theta_sts], axis=1)
    
    peaks = peaks_sts.copy()
    alpha_data = alpha.copy()
    beta_data = beta.copy()
    delta_data = delta.copy()
    gamma_data = gamma.copy()
    theta_data = theta.copy()
    alpha_data_median = alpha_m.copy()
    beta_data_median = beta_m.copy()
    delta_data_median = delta_m.copy()
    gamma_data_median = gamma_m.copy()
    theta_data_median = theta_m.copy()
    alpha_res = channels_alpha_new.copy()
    beta_res = channels_beta_new.copy()
    delta_res = channels_delta_new.copy()
    gamma_res = channels_gamma_new.copy()
    theta_res = channels_theta_new.copy()
    etiketler_data = etiketler.copy()
    kullanici_degerlendirme = participant_ratings.copy()
    video_listesi = video_list.copy()
    return peaks, alpha_data, beta_data, delta_data, gamma_data, theta_data, alpha_data_median, beta_data_median, delta_data_median, gamma_data_median, theta_data_median, alpha_res, beta_res, delta_res, gamma_res, theta_res, etiketler_data, kullanici_degerlendirme, video_listesi
    
    
    
    
    
    


# # NOT: medyan filtresi ile %70 acc//// alpha kanalı ile %89 acc
# from sklearn.model_selection import train_test_split
# x_train, x_test,y_train,y_test = train_test_split(alpha,etiketler,test_size=0.25, random_state=0)
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.transform(x_test)

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# from sklearn.metrics import confusion_matrix
# tahmin4 = gnb.predict(X_test)
# cm3 = confusion_matrix(y_test, tahmin4,labels=gnb.classes_)
# print("\n\n",cm3)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=gnb.classes_)
# disp.plot()
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, tahmin4)*100



# #NOT: medyan filtresi ile birlikte %86 acc/// alpha kanalı ile %100 acc
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
# knn.fit(X_train, y_train)
# tahmin2 = knn.predict(X_test)
# cm1 = confusion_matrix(y_test, tahmin2)
# print(cm1)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=knn.classes_)
# disp.plot()
# accuracy_score(y_test, tahmin2)*100

# import joblib
# filename = 'finalized_model.sav'
# joblib.dump(knn, filename)


# #NOT: alpha kanalı ile %99 acc///medyan filtresi ile %69 acc
# from sklearn.svm import SVC
# svc = SVC(kernel = "rbf")
# svc.fit(X_train, y_train)
# tahmin3 = svc.predict(X_test)
# cm2 = confusion_matrix(y_test, tahmin3)
# print("\n\n",cm2)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=svc.classes_)
# disp.plot()
# accuracy_score(y_test, tahmin3)*100

# import joblib
# filename1 = 'finalized_model_SVM.sav'
# joblib.dump(svc, filename1)


# #NOT: dusuk acc
# from sklearn.model_selection import train_test_split
# x_train1, x_test1,y_train1,y_test1 = train_test_split(var_mean,etiketler,test_size=0.25, random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train1 = sc.fit_transform(x_train1)
# X_test1 = sc.transform(x_test1)

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train1, y_train1)

# from sklearn.metrics import confusion_matrix
# tahmin5 = gnb.predict(X_test1)
# cm4 = confusion_matrix(y_test1, tahmin5,labels=gnb.classes_)
# print("\n\n",cm4)

# from sklearn.metrics import accuracy_score
# accuracy_score(y_test1, tahmin5)*100

# filename2 = 'finalized_model_var.sav'
# joblib.dump(gnb, filename2)



# #NOT: spectral guc yogunlugu ile dusuk acc/// medyan filtresi ile %60 acc gelistirilebilir
# from sklearn.model_selection import train_test_split
# x_train1, x_test1,y_train1,y_test1 = train_test_split(peaks_sts,etiketler,test_size=0.25, random_state=0)
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train1 = sc.fit_transform(x_train1)
# X_test1 = sc.transform(x_test1)

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train1, y_train1)

# from sklearn.metrics import confusion_matrix
# tahmin5 = gnb.predict(X_test1)
# cm4 = confusion_matrix(y_test1, tahmin5,labels=gnb.classes_)
# print("\n\n",cm4)

# from sklearn.metrics import accuracy_score
# accuracy_score(y_test1, tahmin5)*100




