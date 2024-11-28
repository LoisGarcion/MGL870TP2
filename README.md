# MGL870TP2

https://github.com/logpai/loghub

Utilisation de BGL et OpenStack

Pour la partie BGL, il faut mettre le fichier de logs BGL nommé BGL.log dans le dossier BGL.
Il faut ensuite lancer le script processdata.py pour traiter les données.
Puis lancer train.py ou trainFocusRecall.py pour entrainer le modèle.



Pour la partie HDFS, il faut mettre le repertoire HDFS_1 dans /HDFS. Il est important que HDFS_v1/preprocessed/anomaly_label.csv soit présent.
Puis lancer le script tp2.py pour process la data.
Puis train2, pour entrainer.