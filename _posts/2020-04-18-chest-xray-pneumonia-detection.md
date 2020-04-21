---
layout: post
title:  "Detecção de Pneumonia baseado em raios-x"
date:   2020-04-18 23:54:08 -0300
categories: mechine learning
---
# Introdução

Nesse post eu vou falar sobre detecção de pneumonia em pacientes baseando-se na imagem de raio-x do tórax dos mesmos.
Esse recurso já é usado por médicos para fazer a identificação do quadro de pneumonia, entretando, modelos 
computacionais podem ser utilizados para realizar essa tarefa, visando tornar essa detecção
mais acessível para hospitais que não possuam um especialista capacitado para isso e aumentar a escalabilidade. Aqui eu vou abordar modelos baseados em redes neurais, mais
especificamente [redes neurais convolucionais][redes_conv], que vem tornado-se uma técnica muito comum para as tarefas de detecção
e classificação de imagens.
 
No caso da pneumonia, a busca é por áreas distorcidas da imagem causadas pelo impacto do vírus ou bactéria na pulmão do paciente.
Com exemplo, as imagens abaixo mostram o raio-x de dois pacientes, a imagem a esquerda é de um paciente sem pneumonia , enquanto a da direita é de um paciente sem pneumonia. Podemos ver uma diferença notável nas duas imagens, enquanto a imagem a esquerda está mais limpa a da direita parece mais marcada com borrões pretos.
 
![title](https://github.com/jadsonlucio/jadsonlucio.github.io/raw/master/assets/images/chest-xray/normal_chest_image.png)![alt-text-2](https://github.com/jadsonlucio/jadsonlucio.github.io/raw/master/assets/images/chest-xray/pneumonia_chest_image.png)
 
No resto desse artigo eu vou falar sobre como foi o processo de criar um detector de pneumonia. O código completo apresentado nesse artigo está no formato de jupyter notebook e pode ser encontrado no [link][github_code].
 
## Coleta de dados
 
A coleta de dados foi realizada através do site [kaggle][kaggle_link]. O conjunto de dados concentra imagens de raio-x do tórax de pessoas saudáveis e com pneumonia, causada por bactérias ou vírus. Eu utilizei a [api do kaggle][kaggle_api] para baixar os dados que vieram com três pasta, train, test e val, treino, teste e validação. O conjunto de dados possui 5.863 imagens de raio-x dividido em duas categorias Pneumonia e Normal, com 1341 imagens de pacientes sem pneumonia, 1345 imagens de pacientes com pneumonia que foi causada por vírus e 2530 imagens de pacientes com pneumonia que foi causada por bactéria, as outras 624 imagens foram utilizadas para teste.
 
    # comando para instalar o kaggle (Necessário ter o python instalado)
    $ pip install --user kaggle
    # comando para baixar o conjunto de dados (Necessário ter as credenciais)
    $ kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
 
## Exemplo imagens
 
### Sem Pneumonia: 
 
![title](https://github.com/jadsonlucio/jadsonlucio.github.io/raw/master/assets/images/chest-xray/imagens_chest_xray_normal.png)
 
### Com Pneumonia:
 
![title](https://github.com/jadsonlucio/jadsonlucio.github.io/raw/master/assets/images/chest-xray/imagens_chest_xray_pneumonia.png)
 
## Pré-processamento 
 
A primeira parte do pré-processamento foi dividir as imagens em 3 grupos diferentes. O de imagens de pacientes sem pneumonia, o de imagens de pacientes com pneumonia causada por vírus e por fim o grupo de imagens de pacientes com pneumonia causada por bactéria. Após isso as imagens de cada grupo foram redimensionadas para a resolução de 512x512 para padronização, já que muitas das imagens apresentavam divergências de resolução. 
 
## Modelo de previsão usando uma rede neural convolucional (CNN) 
 
Com as imagens dos pacientes normais e com pneumonia o conjunto de dados foi dividido entre treinamento e teste. obs: as imagens de 512x512 foram redimensionadas novamente para 128x128 devido às limitações de hardware, mas isso não deve impactar muito a performance final do modelo.
 
Dois métodos diferentes foram testados, no primeiro a rede deve apenas prever dado uma imagem de raio-x se o paciente tem pneumonia ou não, já no segundo método a rede deve também dizer, caso o paciente possua pneumonia, se ela foi causada por um vírus ou bactéria. Abaixo está a arquitetura da CNN utilizada.
 
{% highlight python %}
import keras
from keras.models import Sequential
from keras.layers import *
 
##model building
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(128, 128, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_category, activation='softmax'))
{% endhighlight %}
 
 
 
 
## Resultado da CNN
 
Os resultados de performance foram obtidos através da previsão dos dados de teste. A figura abaixo apresenta taxa de acerto dos dois modelos criados. Como esperado o modelo responsável só por prever se o paciente tinha pneumonia obteve 75% de acurácia contra 66% do modelo que previa também se a pneumonia foi causada por uma bactéria ou vírus. Isso ocorre porque imagens de pacientes com pneumonia causada por bactéria são muito semelhantes àquelas causadas por vírus, o que torna a previsão muito mais difícil. Não foram realizados testes com outras arquiteturas de CNN ou com outros parâmetros e a arquitetura que foi utilizada foi bem simplificada para não consumir muitos recursos, então provavelmente à acurácia pode ser aumentada com mudança de arquitetura e de parâmetros de treinamento. 


![title](https://github.com/jadsonlucio/jadsonlucio.github.io/raw/master/assets/images/chest-xray/acuracia.png)


[redes_conv]: http://shorturl.at/EHW08
[conv_autoencoders]: http://shorturl.at/fI129
[github_code]: https://github.com/jadsonlucio/Posts-source-code/tree/master/pneumonia%20detection
[kaggle_link]: kaggle.com
[kaggle_api]: https://github.com/Kaggle/kaggle-api