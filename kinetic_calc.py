import pandas as pd
import numpy as np
import os
import logging
import yaml
import plotly.express as px
import matplotlib.pyplot as plt

# Загрузка конфигурации из файла
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pk_data_path = config['pk_data_path']
time_data_path = config['time_data_path']
output_file_path = config['output_file_path']

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(pk_data_path, time_data_path):
    """Загружает данные из указанных файлов Excel."""
    for path in [pk_data_path, time_data_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ошибка: Файл не найден: {path}")

    pk_data = pd.read_excel(pk_data_path, na_values=['NA'], header=0)
    time_data = pd.read_excel(time_data_path)

    # Проверка заголовков
    logging.info(f"Заголовки pk_data: {pk_data.columns.tolist()}")
    logging.info(f"Заголовки time_data: {time_data.columns.tolist()}")

    if pk_data.shape[0] == 0 or time_data.shape[0] == 0:
        raise ValueError("Ошибка: Один из DataFrame пуст.")

    if 'PK1' not in pk_data.columns:
        raise ValueError("Ошибка: В pk_data отсутствует столбец 'PK1'.")

    if time_data.isnull().values.any():
        raise ValueError("Ошибка: Временные интервалы содержат NaN значения.")

    logging.info(f"Загружены данные: pk_data - {pk_data.shape[0]} строк, {pk_data.shape[1]} столбцов; "
                 f"time_data - {time_data.shape[0]} строк, {time_data.shape[1]} столбцов.")
    logging.info(f"Первые 5 строк pk_data:\n{pk_data.head()}")
    logging.info(f"Первые 5 строк time_data:\n{time_data.head()}")

    return pk_data, time_data


def preprocess_data(pk_data, time_data):
    """Предобрабатывает данные, включая преобразование типов."""
    if not isinstance(pk_data, pd.DataFrame) or not isinstance(time_data, pd.DataFrame):
        raise ValueError("pk_data и time_data должны быть экземплярами pandas DataFrame.")

    logging.info(
        f"Количество временных меток: {time_data.shape[1]}, количество столбцов в pk_data: {pk_data.shape[1] - 1}")

    if pk_data.shape[1] > 1:
        pk_data.iloc[:, 1:] = pk_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    nan_counts = pk_data.isnull().sum()
    logging.info(f"Количество NaN значений в каждом столбце после преобразования:\n{nan_counts}")

    nan_handling_method = input("Выберите цифру (метод) обработки NaN значений (1 - fill, 2 - drop, 3 - interpolate): ")

    if nan_handling_method == '1':
        pk_data = pk_data.fillna(0)
    elif nan_handling_method == '2':
        pk_data = pk_data.dropna()
    elif nan_handling_method == '3':
        for col in pk_data.columns[1:]:
            pk_data[col] = pk_data[col].interpolate(axis=0)
    else:
        logging.error("Некорректный метод обработки NaN значений. Используется метод 'fill' по умолчанию.")
        pk_data = pk_data.fillna(0)

    logging.info(
        f"Количество NaN значений в каждом столбце после обработки:\n{pk_data.isnull().sum()}")

    pk_data = pk_data.infer_objects()

    logging.info(f"Первые 5 строк pk_data:\n{pk_data.head(20)}")

    logging.info(f"Типы данных в pk_data после обработки:\n{pk_data.dtypes}")

    return pk_data


def calculate_auc(pk_data, time_data):
    """Расчитывает AUC и возвращает DataFrame с результатами."""
    results = pd.DataFrame({'Subject': pk_data.iloc[:, 0]})
    results['Cmax'] = pk_data.iloc[:, 1:].max(axis=1)
    results['Tmax'] = pk_data.iloc[:, 1:].idxmax(axis=1)

    time_intervals = time_data.iloc[0, :].values  # Предполагаем, что временные интервалы в первом столбце
    auc_values = []

    for index, row in pk_data.iterrows():
        if row.isnull().any():
            logging.warning(f"NaN значение найдено в строке {index}. Пропускаем.")
            auc_values.append(None)
            continue

        # Логирование значений для отладки
        logging.info(
            f"Вычисление AUC для строки {index}: {row[1:].values} с временными интервалами {time_intervals[0:]}")

        # Убедитесь, что длины совпадают
        if len(row[1:]) != len(time_intervals):
            logging.error(f"Ошибка: длины массивов не совпадают для строки {index}: "
                          f"{len(row[1:])} и {len(time_intervals)}.")
            auc_values.append(None)
            continue

        auc = np.trapezoid(row[1:], time_intervals)  # row[1:] - концентрации, time_intervals - временные метки
        auc_values.append(auc)

    results['AUC'] = auc_values
    return results


def save_results(results, output_file_path):
    """Сохраняет результаты в указанный файл Excel."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        with pd.ExcelWriter(output_file_path) as writer:
            results.to_excel(writer, index=False)
        logging.info(f"Результаты успешно сохранены в {output_file_path}.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов: {e}")


def visualize_results(results):
    """Создает визуализацию результатов с помощью Plotly."""
    fig = px.scatter(results, x='Cmax', y='AUC', color='Subject',
                     title='Cmax vs AUC',
                     labels={'Cmax': 'Максимальная концентрация (Cmax)', 'AUC': 'Площадь под кривой (AUC)'},
                     hover_name='Subject')

    fig.update_traces(marker=dict(size=10))
    fig.show()


def visualize_results_matplotlib(results):
    """Создает визуализацию результатов с помощью Matplotlib."""
    plt.figure(figsize=(10, 6))
    plt.scatter(results['Cmax'], results['AUC'], c='blue', alpha=0.5, edgecolors='k')
    plt.title('Cmax vs AUC')
    plt.xlabel('Максимальная концентрация (Cmax)')
    plt.ylabel('Площадь под кривой (AUC)')
    plt.grid(True)

    # Добавление аннотаций для каждого субъекта
    for i, txt in enumerate(results['Subject']):
        plt.annotate(txt, (results['Cmax'].iloc[i], results['AUC'].iloc[i]), fontsize=8, alpha=0.7)

    plt.show()


def main():
    """Основная функция для выполнения анализа."""
    pk_data_path = 'data/pk_data.xlsx'
    time_data_path = 'data/time.xlsx'
    output_file_path = 'data/pharmacokinetic_results.xlsx'

    try:
        pk_data, time_data = load_data(pk_data_path, time_data_path)
        pk_data = preprocess_data(pk_data, time_data)
        results = calculate_auc(pk_data, time_data)

        logging.info("Результаты фармакокинетических измерений:")
        logging.info(results.to_string(index=False))

        average_cmax_after_5th_dose = results['Cmax'].mean()
        logging.info(f"Средняя максимальная концентрация после 5-го введения: {average_cmax_after_5th_dose:.2f} нг/мл")

        save_results(results, output_file_path)

        # Визуализация результатов
        visualize_results(results)

        # Визуализация результатов с помощью Matplotlib
        visualize_results_matplotlib(results)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()