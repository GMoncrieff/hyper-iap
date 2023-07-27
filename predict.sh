

python predict_compare_cuda.py 007kslnc cnn &> output_predict.txt
python predict_raster.py &> output_raster.txt
gdal_translate cnn_class_gpu_int.tif cnn_cuda_cog.tif -of COG -co NUM_THREADS=ALL_CPUS -co COMPRESS=ZSTD &> output_gdal.txt