import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# importação apenas de dados de fácil acesso para construir a árvore
data = pd.read_csv('diabetes.csv', usecols=['Pregnancies', 'Glucose', 'BMI', 'Age', 'Outcome'])
# limpeza dos dados nulos/vazios
data = data.dropna()
# renomear colunas para português para facilitar a visualização e entendimento
data = data.rename(
    columns={'Pregnancies': 'Gravidezes', 'Glucose': 'Glicose', 'BMI': 'IMC', 'Age': 'Idade', 'Outcome': 'Resultado'})

# separando as colunas que irão ser usadas para a árvore de decisão
fatores = ['Gravidezes', 'Glicose', 'IMC', 'Idade']

# monta a árvore de decisão
clf = tree.DecisionTreeClassifier(max_depth=5)  # limitar a 5 nós de profundidade
clf = clf.fit(
    data[fatores],
    data['Resultado']
)


# Função do menu, definir o que fazer
def menu():
    opcao = float(input('Digite 1 para plotar a árvore, e 2 para testa-la! '))
    if opcao == 1:
        plotar_arvore()
    if opcao == 2:
        executar_teste()

# função de plotar a árvore
def plotar_arvore():
    try:
        plt.figure(figsize=(40, 15))
        tree.plot_tree(clf, feature_names=fatores, class_names=['Não', 'Sim'], filled=True, fontsize=10)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('Erro ao fazer plot: ', e)
    finally:
        print('\n Árvore plotada com sucesso! Voltando ao menu. \n ')
        menu()

# função de testar a árvore com dados
def executar_teste():
    print('\nOlá, vamos fazer uma pré-avaliação de diabetes com base em alguns dados, vamos começar\n')
    resultado = ""
    idade = float(input('Qual a sua idade: '))
    peso = float(input('Qual o seu peso: '))
    altura = float(input('Qual a sua altura em cm: '))
    imc = peso / ((altura / 100) ** 2)
    glicose = float(input('Qual o valor da sua última medição de glicose: '))
    gravidezes = float(input('Quantas gravidezes já teve (0 caso não se aplique): '))
    entrada = pd.DataFrame([{
        'Gravidezes': gravidezes,
        'Glicose': glicose,
        'IMC': imc,
        'Idade': idade
    }])
    try:
        resultado = clf.predict(entrada)[0]
    except Exception as e:
        print('Erro ao gerar resultado: ', e)
    finally:
        print('\n Resultado: ',
              '\n Sim, recomendamos consultar um médico \n' if resultado else '\n Não, você não apresenta predisposição, mas em caso de dúvidas procure um médico.\n')
    menu()

### início da exibição no Console e execução do menu pela primeira vez!
print(
    '\nEste é um projeto onde, baseado em dados históricos e dados informados pelo usuário, iremos fazer uma pré-avaliação para diabetes.\n')
menu()
