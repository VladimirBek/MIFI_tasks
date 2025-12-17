# covid-19 xray metadata analytics (pyspark)

Итоговоый проект по инфраструктуре big data: чистка метаданных + spark sql + базовая визуализация.

## структура

```
project/
├── analysis.ipynb
├── presentation.pdf
├── README.md
└── requirements.txt
```

## как запустить локально

1) python 3.9+
2) зависимости:

```bash
pip install -r requirements.txt
```

3) открыть `analysis.ipynb` и запустить все ячейки.

ноут сам скачает `metadata.csv` из репозитория `ieee8023/covid-chestxray-dataset` (github raw).

## заметки

- проект работает только с `metadata.csv`, без скачивания изображений (не нужны для аналитики).
- если запускайте в google colab, можно раскоментить строку `!pip install ...` в первой ячейке.
