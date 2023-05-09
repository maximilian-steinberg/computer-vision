close all; clear all;

% Tabelle mit allen Trainingsdaten laden
labeldata = load("labeldata.mat");

% Durch alle Tabellenzeilen iterieren
for i=1:size(labeldata.labeldata, 1)

    % Dateipfad, Labelname und Polygonlabel auslesen
    imagepath = labeldata.labeldata(i,:).imagepath{:};
    labelname = labeldata.labeldata(i,:).labelname;
    polygon = labeldata.labeldata(i,:).polygon;

    % Mittelpunkt des Polygons
    centre = (min(polygon{:}) + max(polygon{:})) / 2;
    
    % Bild laden, Labelbild aus polygon generieren und uebereinander legen
    im = imread(imagepath);
    label = poly2label(polygon, 1, size(im, 1:2));
    overlay = labeloverlay(im,label);

    % Labelname in Bild einfuegen
    overlay_with_labelname = insertText(overlay, centre, labelname, FontSize=24);

    figure(1)
    imshow(overlay_with_labelname);
    
    pause(0.3)
end