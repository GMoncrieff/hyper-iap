

python predict_compare_cuda.py 007kslnc cnn &> output.txt
python predict_raster.py
gdal_translate cnn_class_gpu_int.tif cnn_cuda_cog.tif -of COG -co NUM_THREADS=ALL_CPUS -co COMPRESS=ZSTD