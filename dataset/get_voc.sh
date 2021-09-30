#!/bin/bash

# Download data
echo "Downloading VOC2007 trainval ..."
curl -LO http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
echo "Downloading VOC2007 test data ..."
curl -LO http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
echo "Downloading VOC2012 trainval ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xf VOCtest_06-Nov-2007.tar
echo "Extracting trainval ..."
tar -xf VOCtrainval_11-May-2012.tar

echo "removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
rm VOCtrainval_11-May-2012.tar

echo "VOC download done."
