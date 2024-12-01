# PharmaKinetics Analyzer

PharmaKinetics Analyzer — это программа для анализа фармакокинетических данных, которая позволяет загружать, обрабатывать и визуализировать данные о концентрации вещества в организме во времени. Программа рассчитывает площадь под кривой (AUC) и максимальную концентрацию (Cmax) на основе введенных данных.


## Использование

1. **Подготовьте данные:**
   Убедитесь, что у вас есть файлы `pk_data.xlsx` и `time.xlsx` в каталоге `data`.

2. **Запустите программу:**
   ```bash
   python kinetic_calc.py
   ```

3. **Следуйте инструкциям на экране:**
   Программа запросит вас выбрать метод обработки NaN значений. Вы можете выбрать один из следующих методов:
   - `1` - Заполнить NaN значения нулями
   - `2` - Удалить строки с NaN значениями
   - `3` - Интерполировать NaN значения

4. **Просмотрите результаты:**
   После завершения анализа результаты будут сохранены в файл `pharmacokinetic_results.xlsx`, а также будут визуализированы с помощью Plotly и Matplotlib.

## Примеры использования

- **Анализ данных о концентрации:**
  Запустите программу с вашими данными, чтобы получить AUC и Cmax для каждого субъекта.

- **Визуализация результатов:**
  Программа автоматически создаст графики, показывающие зависимость между Cmax и AUC, что поможет вам лучше понять фармакокинетические характеристики вещества.

## Примечания

- Убедитесь, что ваши данные соответствуют ожидаемому формату. Примеры структур данных будут доступны в репозитории.
- Программа использует логирование для отслеживания процесса выполнения и возможных ошибок. Логи будут выводиться в консоль.

## Лицензия

Этот проект лицензирован под MIT License - подробности смотрите в файле [LICENSE](LICENSE).
