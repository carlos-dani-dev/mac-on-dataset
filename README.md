## Anotador de datasets automático - MAC
<p>
Este repositório itera o <a href="https://github.com/carlos-dani-dev/classificador-massivo-de-atributos">Classificador Massivo de Atributos</a> sobre qualquer dataset, com qualquer estrutura de pastas.<br>
Este modelo <i>.keras</i> rotula de acordo com 2 atributos, um atributo binário, o <b><i>male_or_not_male</i></b>, e um atributo
multiclasse (6 classes), o <b><i>fitz_type_scale</i></b>.
</p>

<p><br></p>

---
### Requisitos para execução do código fonte
<p>
Para baixar os frameworks usados, cada um em sua versão exata, rode no terminal:<br>
  <b><i>pip install -r requirements.txt</i></b>
</p>

<p><br></p>

---
### Pipeline de execução do código fonte
<p>
1) Primeiro vetorize o banco de imagens, rodando o script <b><i>loader.py</i></b>, escrevendo no terminal:<br>
  <b><i>python loader.py</i></b>
</p>
<p>
2) Rotule o banco de imagens vetorizado, rodando o script <b><i>annotator.py</i></b>, escrevendo no terminal:<br>
  <b><i>python annotator.py</i></b>
</p>
