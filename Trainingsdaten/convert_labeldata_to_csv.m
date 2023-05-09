% Konvertiere 'labeldata.m' in eine '.csv' datei

% Bereinige Matlab-Umgebung
close all; clear all;

% Lese die manipulierten Label-Informationen
struct = load("altered_labeldata.mat");

% Lese die Label-Daten aus dem Struct
labeldata = getfield(struct,"labeldata");

% Ordner identifizieren
currentFolder = pwd;

% Dateinamen setzen
fileName = strcat(currentFolder, 'labeldata.csv');

% Schreibe Label-Data in '.csv'
writetable(labeldata,fileName)