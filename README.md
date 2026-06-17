## Anotador de datasets automático - MAC
<p>
Este repositório itera o <a href="">Classificador Massivo de Atributos definido</a> sobre qualquer dataset, com qualquer estrutura de pastas.<br>
Este modelo <i>.keras</i> rotula de acordo com 2 atributos, um atributo binário, o <b><i>male_or_not_male</i></b>, e um atributo
multiclasse (6 classes), o <b><i>fitz_type_scale</i></b>.
</p>

### Requisitos para execução do código fonte


### Pipeline de execução do código fonte
<p>
1) Primeiro vetorize o banco de imagens, rodando o script <b><i>loader.py</i></b>, escrevendo no terminal:<br>
  <b><i>python loader.py</i></b>
</p>
<p>
2) Rotule o banco de imagens vetorizado, rodando o script <b><i>annotator.py</i></b>, escrevendo no terminal:<br>
  <b><i>python annotator.py</i></b>
</p>
