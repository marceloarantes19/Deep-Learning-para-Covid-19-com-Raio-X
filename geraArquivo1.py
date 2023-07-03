import csv
class GeraArquivo:
  def geraHistorico(nome, h):
    f = open(nome, 'w', newline='', encoding='utf-8')

    # 2. cria o objeto de gravação
    w = csv.writer(f)
    w.writerow(["accuracy", "val_accuracy", "loss", "val_loss"])

    # 3. grava as linhas
    for i in range(len(h["accuracy"])):
      w.writerow([h["accuracy"][i], h["val_accuracy"][i], h["loss"][i], h["val_loss"][i]])

    # Recomendado: feche o arquivo
    f.close()

  def geraPredicao(nome, categorias, preditos):
    f = open(nome, 'w', newline='', encoding='utf-8')

    # 2. cria o objeto de gravação
    w = csv.writer(f)
    w.writerow(["real", "predito"])

    # 3. grava as linhas
    for i in range(len(categorias)):
      w.writerow([categorias[i], preditos[i]])

    # Recomendado: feche o arquivo
    f.close()