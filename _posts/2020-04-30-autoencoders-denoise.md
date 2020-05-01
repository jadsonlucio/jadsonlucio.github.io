---
layout: post
title:  "Autoencoders - remoção de ruidos"
date:   2020-04-30 23:54:08 -0300
categories: mechine learning
---


# Introdução 

[Autoencoder][autoencoders] é um tipo de arquitetura de rede neural que possui duas partes. O encoder, que recebe a entrada, e fica responsável por reduzir gradualmente a dimensão sua dimensão, e o decoder que é responsável por receber uma entrada em menor dimensão e aumentá-la gradualmente. fazendo uma analogia com imagens, o encoder vai diminuir a resolução dessa imagem e o decoder vai aumentá-la. Então o objetivo do autoencoder é reproduzir a entrada.
 
Mas qual o sentido de prever a mesma coisa? O segredo dos autoencoders está no processo de redução de dimensionalidade, já que a rede neural deve encontrar uma forma de representar os dados em uma menor dimensão (algo parecido com o que o PCA faz), é esse tipo de característica que impede o autoencoder de apenas aprender uma função identidade. Dito isso, existem várias aplicações para os autoencoders, as mais comuns são: Geração de conteúdo, Detecção de anomalias e remoção de ruídos.

Nesse post eu vou abordar a remoção de ruído utilizando autoencoders, mais precisamente, a remoção de ruídos em imagens. Ruído este que pode ser produto de um redimensionamento ou problema na camera fotografica. Para realizar essa tarefa, os dados de treinamento do autoencoder passam por um processo de inserção de ruído e após isso o autoencoder e treinando para ter como entrada uma imagem com ruído e obter como saída essa imagem sem ruído. Eu também vou comparar o resultado da remoção de ruído utilizando o autoencoder contra dois dos algoritmos mais básicos para resolver essa tarefa. Os filtros de média e mediana. O filtro da média consiste em para cada pixel da imagem tirar a média dos pixels ao redor, já o da mediana é quase igual, com a exceção de que se tira
a mediana dos pixels ao redor. Uma desvantagem do autoencoder é que ele é específico de domínio, ou seja, enquanto os filtros de média e mediana podem ser usados em qualquer imagem o autoencoder só pode ser usado em imagens do domínio que ele foi treinado. Esse problema pode ser contornado utilizando imagens de vários domínios mas os resultado não são tão bons. Já uma desvantagens dos filtros de média e mediana é que eles não se saem bem em imagens coloridas. 
 
O código utilizado para obtenção dos resultados está na minha [página][link_do_codigo] do github, ele está no formato de júpiter notebook.

# Conjunto de dados e pré-processamento

O conjunto de imagens utilizado para esse experimento foi o de faces de celebridades disponível em http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, que contem cerca de 204.000 imagens frontais de celebridades.


![alt text](/assets/images/autoencoder_denoise/faces.png)

Após o download e a descompactação, 4000 imagens foram carregadas e redimensionada para a resolução de 28x28 pixels (devido a limitações de hardware). 
 
A próxima etapa é aplicar o ruído nas imagens, a fim de simular o processo real. Existem várias formas de se fazer isso, nesse post vão ser abordados quatro dessas formas, o ruído gaussiano, o de poisson, o salt and pepper e o multiplicativo. O ruído gaussiano é a aplicação de uma máscara em toda a imagem, essa máscara possui dados com uma distribuição gaussiana. O método de poisson funciona de forma semelhante ao gaussiano mas com a máscara obedecendo uma distribuição de poisson. O ruído de salt and pepper busca trocar a cor de pixels randômicos na imagem. Já o ruído multiplicativo aplica uma máscara à imagem, que é produto de um ruído uniforme com média e variância definidos, multiplicado pela imagem. A código abaixo mostra a implementação dos algoritmos que aplicam esses ruídos.



{% highlight python %}
import numpy as np

# Ruido aditivo Gaussiano
row,col,ch= image.shape
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
noisy = image + gauss

# Ruido aditivo de poisson
vals = len(np.unique(image))
vals = 2 ** np.ceil(np.log2(vals))
noisy = np.random.poisson(image * vals) / float(vals)

# Ruido salt and pepper
row,col,ch = image.shape
s_vs_p = 0.5
amount = 0.004
out = np.copy(image)

num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
out[coords] = 1


num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
out[coords] = 0

# Ruido multiplicativo
row,col,ch = image.shape
gauss = np.random.randn(row,col,ch)
gauss = gauss.reshape(row,col,ch)        
noisy = image + image * gauss
{% endhighlight %}

A imagem abaixo mostra a aplicação desses quatro ruídos em uma imagem do conjunto de dados.

![alt text](/assets/images/autoencoder_denoise/noise_methods.png)


# Treinamento do autoencoder

Para o treinamento os dados de entrada,com ruído, e de saída, sem, foram divididos entre treino e teste, em uma proporção de 
75% para treino e 25% para teste.
A arquitetura do autoencoder utilizado nesse post está mostrada abaixo:

        Model: "encoder"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         [(None, 28, 28, 3)]       0         
        _________________________________________________________________
        conv2d (Conv2D)              (None, 14, 14, 32)        896       
        _________________________________________________________________
        leaky_re_lu (LeakyReLU)      (None, 14, 14, 32)        0         
        _________________________________________________________________
        batch_normalization (BatchNo (None, 14, 14, 32)        128       
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 7, 7, 64)          18496     
        _________________________________________________________________
        leaky_re_lu_1 (LeakyReLU)    (None, 7, 7, 64)          0         
        _________________________________________________________________
        batch_normalization_1 (Batch (None, 7, 7, 64)          256       
        _________________________________________________________________
        flatten (Flatten)            (None, 3136)              0         
        _________________________________________________________________
        dense (Dense)                (None, 100)               313700    
        =================================================================
        Total params: 333,476
        Trainable params: 333,284
        Non-trainable params: 192
        _________________________________

        Model: "decoder"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_2 (InputLayer)         [(None, 100)]             0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 3136)              316736    
        _________________________________________________________________
        reshape (Reshape)            (None, 7, 7, 64)          0         
        _________________________________________________________________
        conv2d_transpose (Conv2DTran (None, 14, 14, 64)        36928     
        _________________________________________________________________
        leaky_re_lu_2 (LeakyReLU)    (None, 14, 14, 64)        0         
        _________________________________________________________________
        batch_normalization_2 (Batch (None, 14, 14, 64)        256       
        _________________________________________________________________
        conv2d_transpose_1 (Conv2DTr (None, 28, 28, 32)        18464     
        _________________________________________________________________
        leaky_re_lu_3 (LeakyReLU)    (None, 28, 28, 32)        0         
        _________________________________________________________________
        batch_normalization_3 (Batch (None, 28, 28, 32)        128       
        _________________________________________________________________
        conv2d_transpose_2 (Conv2DTr (None, 28, 28, 3)         867       
        _________________________________________________________________
        activation (Activation)      (None, 28, 28, 3)         0         
        =================================================================
        Total params: 373,379
        Trainable params: 373,187
        Non-trainable params: 192

Varios autoencoders foram treinados, um para cada dataset produzido com um método de ruído expecifico. A figura abaixo mostra o resultado de cada um desses. 

![alt text](/assets/images/autoencoder_denoise/auto_encoders_result.png)

Os resultados mostram que as imagens previstas pelos autoencoders apresentaram borrões semelhantes ao que é visto com o filtro de média, um dos motivos disso, além da arquitetura simples utilizada para construir o autoencoder, está no fato de que a função de erro que o autoencoder tenta minimizá que é a mse (Mean square error) não é muito recomendada para utilizar em imagens, já que na maioria das vezes ela induz a geração imagens com efeitos de borrão. Um alternativa a isso é utilizar uma rede neural para calcular esse erro, dessa forma ela vai criar uma função de erro personalizada para o domínio específico do treinamento. Esse é o conceito das redes neurais adversariais (GAN), que possui uma rede chamada de generator que é responsável pela saída da rede, e outra rede chamada de discriminator que vai agir como uma função de erro, verificando se a saída do gerador está dentro da distribuição do conjunto de treinado. Na sessão Bônus eu vou usar uma GAN para realizar a tarefa de remoção de ruído e nós vamos comparar o resultado contra o autoencoder.

# Resultados

O gráfico abaixo apresenta os erros dos respectivos métodos de filtragem utilizados. Podemos ver que o método utilizando autoencoder se saiu pior em todas as situações.

![alt text](/assets/images/autoencoder_denoise/results_absolute_error.png)

A função de erro utilizada foi o erro absoluto, que de acordo com a formula abaixo, compara (subtrai) duas imagens pixel a pixel tira o valor absoluto dessa subtração e depois soma cada um dos pixels em um resultado final que é divido pela quantidade de pixels da imagem.

{% highlight python %}
sum(abs(ImageA - ImageB)/pixel_quant)
{% endhighlight %}

Mas nem sempre a função de erro está de acordo com os resultados do ponto de vista humana. Partindo desse princípio eu fiz uma avaliação manual dos resultados e o autoencoder visualmente se saiu melhor em todos os casos analisados. A figura abaixo mostra alguns desses resultados.

![alt text](/assets/images/autoencoder_denoise/results_images.png)

# Bônus: GANs

[Redes neurais adversariais][gans] tem ganhado cada vez mais atenção ao longo dos anos devido a sua alta capacidade de geração de dados de qualidade que são muito semelhantes ao conjunto de treinamento. Mas essa não é a sua única aplicação, elas podem ser usadas para: [Transferência de estilo][gans_style_transfer], [Pintura de imagem][gans_image_inpaint], [Aumento de resolução (Superesolução)][gans_superesolution] entre outras inúmeras tarefas. 
 
Também é possível realizar remoção de ruído. Existem várias arquiteturas diferentes de GANs para realizar essa tarefa. Nesse post eu vou utilizar a arquitetura conhecida como pix2pix, que recebe uma imagem e tenta traduzir essa imagem para um outra imagem em outro contexto. Nesse caso a tradução vai ser de uma imagem com ruído para uma sem ruído.

A imagem abaixo mostra o resultado de algumas imagens que tiveram o ruido removido pela GAN:

![alt text](/assets/images/autoencoder_denoise/gan_denoise.png)

Os resultados mostram que utilizando a GAN, o processo de ruído não apresentou os borrões típicos do autoencoder. Com relação ao erro absoluto as GANs também apresentaram uma melhora significativa com relação aos autoencoders. A figura abaixo mostra o erro média absoluto calculado para o processo de remoção de ruído. No ruído do tipo salt and pepper essa melhora chegou a ser de 1200% em valores absolutos.
 
![alt text](/assets/images/autoencoder_denoise/pix2pix_results.png)

[autoencoders]: https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_0436.pdf
[gans]: https://arxiv.org/pdf/1511.06434.pdf%C3
[gans_style_transfer]: https://arxiv.org/abs/1703.10593
[gans_image_inpaint]: https://arxiv.org/abs/1703.05192
[gans_superesolution]: https://arxiv.org/abs/1609.04802
[link_do_codigo]: https://github.com/jadsonlucio/Posts-source-code/tree/master/Autoencoders%20denoising