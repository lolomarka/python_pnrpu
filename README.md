# Проект по дисциплине "Программирование (PYTHON)"

Проект является набором инструментов для обучения классификатора клеток in-vitro пробы "Жидкости человека", разрабатываемого в ходе выполнения магистерской диссертации.

## Описание содержимого репозитория
В репозитории находятся 3 скрипта:
### model_trainer.py
Код для обучения модели. Его содержимое не оформлено как command-line-utility, потому что скрипт постоянно изменяется, если оформить его как утилиту, это сильно усложнит дальнейшую работу над скриптом.

Обучение текущей модели на датасете более 1 млн картинок, разделённом на train/test/val с соотношением 0.5/0.25/0.25 и использованием 25 эпох дал результат расчётной точности 84%.
### split.py
Сommand-line утилита, для подготовки выборки для обучения.
Имеет следующий синтаксис:
```bash
split.py [-h] -i[--input] <INPUT_PATH> -o[--output] <OUTPUT_PATH> --train <TRAIN_RATIO> --test <TEST_RATIO> --validation <VALIDATION_RATIO>
```   
Делит выборку из INPUT_PATH на train,test,val и помещает в OUTPUT_PATH.
Сумма TRAIN_RATIO + TEST_RATIO + VALIDATION_RATIO должна равняться 1.

### classificator.py
Command-line утилита, для проверки работы классификатора.
Имеет следующий синтаксис:
```bash
classificator.py -i[--input] <INPUT_PATH> -o[--output] <OUTPUT_PATH> -m[--model] <MODEL_PATH>
```
- INPUT_PATH - директория с изображениями для классификации (по-умолчанию `./input`);
- OUTPUT_PATH - директория назначения, для классификации (по-умолчанию `./output`);
- MODEL_PATH - путь до файла модели (по-умолчанию `./model.keras`) 

В output_path создаются 10 подпапок, соответствующие классам.
- ART
- EOS
- LIN
- LYM
- MAC
- MON
- NEU
- NRBC
- PLA
- RBC

## Советы по работе
Все необходимые (используемые) библиотеки оформлены в `requirements.txt`, для начала работы советую определить и активировать venv:
```bash
python3 -m venv env
source ./env/bin/activate
```
И установить все необходимые библиотеки
```bash
pip3 install -r requirements.txt
```

### Примечание
Обучение модели и тестирование работы классификатора происходит на Windows машине с GPU. Используется tensorflow 2.15, работа происходит из под WSL. При желании работать на CPU (не используется в работе потому что используемый dataset для обучения содержит >1 млн картинок 249x249), можно установить просто tensorflow и split-folders.