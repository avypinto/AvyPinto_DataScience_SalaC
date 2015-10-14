
# coding: utf-8

# #Encontro 10
# 
# **Preparo Prévio:**
# 1.	Leitura prévia necessária: Magalhães e Lima (7ª. Edição): Modelo Normal (pág. 197 a 203).
# 
# **Hoje:**
# 1.	Descrever modelos contínuos quanto aos resultados teóricos.
# 2.	Explicar a utilização de modelos probabilísticos no contexto da literatura estatística.
# 3.	Contrastar resultados teóricos e empíricos.
# 4.	Fechamento do conteúdo.
# 
# **Próxima aula:**
# 1.	Leitura prévia necessária: Magalhães e Lima (7ª. Edição): Seção 5.1 (pág. 137) e Seção 5.2 (pág. 146).
# 
# **Atenção:**
# <font color=red> Algumas respostas numéricas deste *IPython notebook* deverão ser inseridas na seção adequada do *Blackboard*</font>
# ___

# In[12]:

get_ipython().magic('matplotlib inline')
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from numpy import linspace


# Nessa aula, teremos contato com algumas distribuições contínuas e como essas podem ser utilizadas para modelagem de dados reais.
# 
# **Distribuição Normal**
# 
# A distribuição Normal tem um abrangente uso prático na modelagem de diversos tipos de variáveis, como, por exemplo, preços, notas, alturas e retornos de ações; entretanto, seu principal destaque está na inferência estatística.
# 
# Teoricamente, a função densidade de probabilidade (FDP) de uma distribuição Normal com parâmetros $\mu \in \mathbb{R}$ e $\sigma^2 > 0$ definida por:
# 
# $$
# f(x|\mu, \sigma^2) = f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
# \quad\quad\quad\quad
# (1)
# $$
# 
# com $x \in \mathbb{R}$. Quando uma variável $X$ tem distribuição Normal com tal parametrização, pode-se afirmar que sua esperança e variância são dadas, respectivamente, por $E(X) = \mu$ e $\mathrm{Var}(X) = \sigma^2$. Como notação, utiliza-se $X \sim N(\mu, \sigma^2)$ e lê-se: “a variável $X$ segue uma distribuição Normal com média $\mu$ e variância $\sigma^2$”.
# 
# A Figura 1 mostra a distribuição Normal para alguns valores do parâmetro $\mu = 10;20;30$ e de $\sigma^2=25;100;400$. O primeiro gráfico mostra que o aumento dos valores de $\mu$ deslocam a curva para direita e diminuí-los, deslocam a curva para esquerda. O segundo gráfico mostra o efeito da mudança de $\sigma^2$, deixando visível que aumentar o desvio padrão achata a curva e diminuí-lo deixa a curva estreita.
# 
# <img src="Normal1.png"/>
# <img src="Normal2.png"/>
# <center><sup>**Figura 1.** Função densidade de probabilidade da Normal
# sob efeito das mudanças de $\mu$ e de $\sigma^2$.</sup></center>
# 
# De maneira geral, para calcular a probabilidade da variável contínua $X$ não exceder um valor $k$, é necessário obter a área abaixo da curva até esse ponto, isto é, calcular por integral a $P(X \leq k)$. Entretanto, se $X$ segue uma distribuição Normal, calcular essa probabilidade não é trivial por causa da complexidade de integrar a função descrita na equação (1). Não é de hoje que experimentos vêm sendo modelados por essa distribuição e para conseguir extrair as probabilidades desejadas, alguns recursos e propriedades dessa distribuição foram desenvolvidos.
# 
# Com o próximo exercício, você deve ser capaz de citar as propriedades da distribuição normal, saber padroniza-la para uma única distribuição conhecida como distribuição normal padrão e ainda saber aplicá-la para tomada de decisão.

# 1) Considere uma variável $X$ com distribuição normal e com alguns valores para média $\mu$ e variância $\sigma^2$, ou seja, $X \sim N(\mu, \sigma^2)$. Para cada caso, responda:
# 
# 1.1. Qual a probabilidade de ocorrência das seguintes situações?
# 
#  a. $X \sim N(10, 100) \Rightarrow P(X < 0); P(X > 20)$;
#  
#  b. $X \sim N(10, 100) \Rightarrow P(X < -10); P(X > 30)$;
# 
#  c. $X \sim N(50, 100) \Rightarrow P(X < 40); P(X > 60);$;
#  
#  d. $X \sim N(50, 100) \Rightarrow P(X < 30); P(X > 70)$;
#  
#  e. $X \sim N(0, 25) \Rightarrow P(X < -5); P(X > 5)$;
#  
#  f. $X \sim N(0, 25) \Rightarrow P(X < -15); P(X > 15)$;
#  
#  g. $X \sim N(20, 25) \Rightarrow P(X < 15); P(X > 25)$;
#  
#  h. $X \sim N(20, 25) \Rightarrow P(X < 5); P(X > 35)$;
# 
# 
# 

# Compare os resultados. Utilizando o desenho da distribuição normal, descreva as propriedades que envolvem os valores de $\mu$ e $\sigma^2$. 
# 
# <sup>Dica: as duas primeiras probabilidades descritas acima podem ser obtidas com os seguintes comandos no Python:</sup>

# In[3]:

stats.norm.cdf(0, loc=10, scale=10)


# In[4]:

1 - stats.norm.cdf(20, loc=10, scale=10)


# In[38]:

#Resolvendo dos dois modos

print("b) A probalibidade é ",stats.norm.cdf(-10, loc=10, scale=10))
print("b) A probalibidade é", 1 -stats.norm.cdf(30, loc=10, scale=10))

print("c) A probalibidade é" ,stats.norm.cdf(40, loc=50, scale=10))
print("c) A probalibidade é" ,1 -stats.norm.cdf(60, loc=50, scale=10))

print("d) A probalibidade é ",stats.norm.cdf(30, loc=50, scale=10))
print("d) A probalibidade é ",1 -stats.norm.cdf(70, loc=50, scale=10))

print("e) A probalibidade é ",stats.norm.cdf(-5, loc=0, scale=5))
print("e) A probalibidade é ",1 -stats.norm.cdf(5, loc=0, scale=5))

print("f) A probalibidade é ",stats.norm.cdf(-15, loc=0, scale=5))
print("f) A probalibidade é ",1 -stats.norm.cdf(15, loc=0, scale=5))

print("g) A probalibidade é ",stats.norm.cdf(15, loc=20, scale=5))
print("g) A probalibidade é ",1 -stats.norm.cdf(25, loc=20, scale=5))



print("h) A probalibidade é ",stats.norm.cdf(5, loc=20, scale=5))
print("h) A probalibidade é ",1 -stats.norm.cdf(35, loc=20, scale=5))


# 1.2. Ache, para $X \sim N(20, 25)$:
# 
# **a.** $P(X < 13)$
# 
# **b.** $P(X > 27)$ 
# 
# **c.** Como se fazia para obter tais probabilidades numa época em que o computador não era de fácil acesso?

# In[39]:

stats.norm.cdf(13,loc=20,scale=5)


# In[40]:

1 - stats.norm.cdf(27,loc=20,scale=5)


# >Para o calculo de probabilidades numa época em que o computador não era de fácil acesso usava-se a "Transformaçao em Z"  e depois se realizava uma comparação com a tabela

# ___
# 2) **Problema de gerenciamento de estoque:**
# 
# **2.1.** Uma loja de produtos automotivos vende um certo lubrificante. Sempre que o estoque chega a 21 litros um novo pedido de compra é feito. 
# 
# O gerente da loja acha que a quantidade deixada em estoque até que o pedido do lubrificante seja entregue pode estar prejudicando suas vendas por medo de faltar produto.
# 
# Verificou-se que durante o período entre o pedido de compra e a entrega vende-se, em média, 15 litros de lubrificante, com desvio padrão de 6 litros. Verificou-se também que a distribuição da demanda neste período é bem aproximada por uma normal. O valor deixado em estoque no momento do pedido é adequado? 
# 
# 
# 

# In[43]:

1-stats.norm.cdf(21, loc=15,scale=6)


# > Não é adequado, pois sobre essas condições existe uma chance de 15% do estoque acabar antes que o novo pedido chegue. Seria aconselhavel mudar a politica de encomenda.
# 

# **2.2.** O gerente pretende demorar mais tempo até fazer novos pedidos de compra. Logo, decide que um novo pedido de compra será feito sempre que o estoque chegar a 20 litros.
# 
# Calcule agora a probabilidade de que o estoque acabe antes que o pedido chegue à loja

# In[21]:

1 - stats.norm.cdf(20, loc=15,scale=6)


# 
# 
# **2.3.** Com quantos litros de lubrificante no estoque a loja de produtos automotivos deve fazer o pedido de compra de modo a ter no máximo 5% de probabilidade de ficar sem lubrificante?

# In[47]:

1 - stats.norm.cdf(24.87, loc=15,scale=6)
# Nesse caso, aproximadamente 24.9 litros


# ___
# 3) Uma empresa deve decidir em qual região construir uma padaria: **Bairro A** ou **Bairro B**.
# 
# > **Como escolher a localização do seu negócio**
# 
# > *Uma das primeiras (e mais importantes) decisões que se deve tomar ao ao abrir o negócio próprio, é escolher a localização do mesmo, pois trata-se de um ponto estratégico para o sucesso do empreendimento.*
# 
# > <sup>Fonte: http://guiadoempreendedor.net/como-escolher-a-localizacao-do-seu-negocio/</sup>

# Os investidores visam atingir um público alvo de maior renda familiar acreditando que essa seja uma boa *proxy* para medida de sucesso do negócio. Logo, a decisão final deverá ser baseada em informações sobre a renda dos moradores dessas regiões. Em princípio, quanto maior o número de famílias com rendas mais altas, maior será a chance do empreendimento ser bem sucedido e nisso, os dois bairros são igualmente populosos. Compare as distribuições de renda pessoal (em salários mínimos) dos dois bairros. Qual é o melhor bairro para construir a padaria? Justifique.
# 
# * Faça uma breve análise descritiva
# 
# * Verifique, graficamente, se a distribuição normal pode ser considerada um bom ajuste a variável renda, segmentado por bairro. Use a média amostral e desvio padrão amostral de cada bairro como parâmetros estimados das distribuições normais.
# 
# * Considerando o modelo normal ajustado, tome uma decisão de escolha de bairro. Saiba justificar adequadamente sua resposta.

# In[11]:

Rendas = pd.read_table('Rendas.txt')
Rendas


# In[15]:

Rendas.describe()
mediaRA = Rendas.A.mean()
StdA = Rendas.A.std()
x = linspace (4,10,61)
pd.DataFrame.plot(kind='hist', data=Rendas.A, bins=range(0, 10), normed=True, legend=False)
plt.plot(x, stats.norm.pdf(x,loc=mediaRA,scale=StdA), '-', color='red')
plt.legend(["Normal Teorica em A","Normal em A"], loc = "upper left")


# In[48]:

mediaRB = Rendas.B.mean()
StdB = Rendas.B.std()
x = linspace (4,10,61)
pd.DataFrame.plot(kind='hist', data=Rendas.B, bins=range(0, 10), normed=True, legend=False,color="orange")

plt.plot(x, stats.norm.pdf(x,loc=mediaRB,scale=StdB), '-', color='magenta')
plt.legend(["Normal Teorica em B","Normal em B"], loc = "upper left")


# In[17]:

plt.plot(x, stats.norm.pdf(x,loc=mediaRB,scale=StdB), '-', color='magenta')
plt.plot(x, stats.norm.pdf(x,loc=mediaRA,scale=StdA), '-', color='red')
plt.legend(["Normal em B","Normal em A"], loc = "upper left")


# > A partir da análise dos parâmetros, o bairro A aparenta ser de maior interesse para os investidores. Considerando que o consumo no empreedimento deve ser bastante homogêneo e nao depende muito da renda do consumidor (rendas maiores que o suficiente nao devem influenciar o quanto a pessoa gasta na loja,uma vez o que interessa ao empreendedor é um lugar onde há mais pessoas com a renda suficiente para gastar o que precisam para a loja ser rentável.Por exemplo: 2 pessoas gastando 100Rs e oito podendo gastar apenas 20Rs, rendem menos do que 10 pessoas podendo gastar 40Rs
# Sendo assim, o Bairro A demonstra possuir mais consumidores dentro da faixa esperada, pois sua media é maior quando comparado ao bairro B. 

# In[ ]:



