.PHONY: data geo weather trips crashes

data: geo weather trips crashes
	@echo "All datasets downloaded/built into raw/interim."

geo:
	python -m src.data.download_borough_geojson

bike_counts:
	python -m src.data.download_bike_counts

weather:
	python -m src.data.download_weather

trips:
	python -m src.data.download_tripdata

crashes:
	python -m src.data.download_crashdata
