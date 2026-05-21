# Projeto: Identificador de Ratas — Versão Client-Side

Guia completo para rodar o projeto **sem backend**, usando apenas arquivos estáticos,
TensorFlow.js no navegador e IndexedDB para persistência local.

---

## Como esta versão difere do guia original

| Aspecto | Versão original | Esta versão |
|---|---|---|
| Servidor | Node.js + Express | Nenhum (só arquivos estáticos) |
| Inferência | `@tensorflow/tfjs` | `@tensorflow/tfjs` (WebGL no navegador) |
| Banco de dados | SQLite | IndexedDB (no próprio navegador) |
| Imagens salvas | Pasta `uploads/` | Blobs no IndexedDB |
| Histórico | Acessível de qualquer dispositivo | Apenas no navegador onde rodou |
| Hospedagem | Servidor Node rodando | `npx serve`, GitHub Pages ou Netlify |

**Limitação principal:** o histórico de capturas fica preso naquele navegador específico.
Se o celular captura e o computador exibe o dashboard, eles não sincronizam automaticamente.
Isso é aceitável se você usar um único dispositivo, ou se o objetivo for aprender.

---

## Visão geral do fluxo

```
Celular (câmera)
    → detecta movimento em JS
        → captura frame do canvas
            → TensorFlow.js (no próprio navegador)
                → identifica a rata
                    → salva em IndexedDB
                        → dashboard lê o IndexedDB e lista as aparições
```

O modelo **não é treinado no navegador** — isso seria lento e instável.
Você treina uma única vez (usando uma das duas opções desta fase) e carrega o
arquivo `model.json` resultante nos HTMLs.

---

## Estrutura de pastas

```
projeto-ratas/
├── modelo/
│   ├── model.json          ← gerado pelo treinamento
│   └── weights.bin         ← pesos do modelo (gerado junto com model.json)
├── dataset/                ← usado só durante o treinamento, não vai para produção
│   ├── cinzenta/
│   ├── labradora/
│   ├── nininha/
│   └── bibranca/
├── camera.html             ← detector de movimento + inferência
├── dashboard.html          ← histórico de aparições
└── db.js                   ← módulo IndexedDB (importado pelos dois HTMLs)
```

Não há `package.json` de produção. A pasta `dataset/` e os scripts de treinamento
são ferramentas de desenvolvimento — não fazem parte do que você hospeda.

---

## Fase 1 — Coleta do dataset

Esta fase é idêntica ao guia original. O objetivo é acumular fotos rotuladas das 4 ratas.

### Como coletar sem servidor

Você ainda precisa de um jeito de receber as fotos do celular. A solução mais simples:
use o servidor Node.js **só nesta fase**, apenas como receptor de uploads —
sem modelo, sem banco, só salvando os arquivos em disco.

**Servidor mínimo de coleta (`coletor.js`):**

```js
// Instale: npm install express multer cors
const express = require('express');
const multer  = require('multer');
const cors    = require('cors');
const path    = require('path');

const app  = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.static('.')); // serve camera.html

app.post('/upload', upload.single('imagem'), (req, res) => {
  console.log('Foto recebida:', req.file.filename);
  res.json({ ok: true });
});

app.listen(3000, () => console.log('Coletor rodando em http://localhost:3000'));
```

Execute com `node coletor.js`, deixe rodar por algumas noites e depois mova
manualmente cada foto para a pasta correta dentro de `dataset/`.

**Meta:** 80–150 fotos por rata, com variedade de ângulos e iluminação.

```
dataset/
  cinzenta/           → 80–150 fotos
  labradora/   → 80–150 fotos
  nininha/    → 80–150 fotos
  bibranca/          → 80–150 fotos
```

Após a coleta, o servidor Node não é mais necessário.

---

## Fase 2 — Treinamento do modelo

Esta é a fase central desta versão. O resultado sempre é um arquivo `model.json`
(mais um ou mais `.bin` de pesos) que o navegador vai carregar.

Escolha **uma** das duas opções abaixo.

---

### Opção A — Script Node.js local

**Quando usar:** você tem Node.js instalado e prefere não depender de nuvem.
O treinamento roda na sua máquina, usando a GPU via bindings nativos do TensorFlow.

#### A.1 — Preparar o ambiente

Crie uma pasta de trabalho separada para o treinamento:

```
treinamento/
├── treinar.js
├── dataset/        ← copie ou faça symlink do dataset principal
└── package.json
```

Instale as dependências:

```bash
cd treinamento/
npm init -y
npm install @tensorflow/tfjs sharp
```

**Por que `@tensorflow/tfjs` aqui?** Esta versão usa bindings C++ nativos e é
muito mais rápida que a versão browser. Ela só é necessária para treinar — a inferência
no navegador usa a versão leve (`@tensorflow/tfjs`).

#### A.2 — Script de treinamento (`treinar.js`)

```js
const tf    = require('@tensorflow/tfjs');
const sharp = require('sharp');
const fs    = require('fs');
const path  = require('path');

// Configuração
const CLASSES      = ['cinzenta', 'labradora', 'nininha', 'bibranca'];
const DATASET_DIR  = './dataset';
const IMG_SIZE     = 224;
const EPOCHS       = 30;
const BATCH_SIZE   = 32;
const SAIDA_MODELO = 'file://./modelo-exportado';

// ─── Carregamento de imagens ────────────────────────────────────────────────

async function carregarImagem(caminho) {
  const buffer = await sharp(caminho)
    .resize(IMG_SIZE, IMG_SIZE)
    .removeAlpha()
    .raw()
    .toBuffer();

  return tf.tensor3d(buffer, [IMG_SIZE, IMG_SIZE, 3], 'float32').div(255);
}

async function carregarDataset() {
  const imagens = [];
  const labels  = [];

  for (let i = 0; i < CLASSES.length; i++) {
    const pasta = path.join(DATASET_DIR, CLASSES[i]);
    const arquivos = fs.readdirSync(pasta)
      .filter(f => /\.(jpg|jpeg|png)$/i.test(f));

    console.log(`Carregando ${CLASSES[i]}: ${arquivos.length} imagens`);

    for (const arquivo of arquivos) {
      const tensor = await carregarImagem(path.join(pasta, arquivo));
      imagens.push(tensor);
      labels.push(i);
    }
  }

  // Embaralhar pares (imagem, label) juntos
  const indices = tf.util.createShuffledIndices(imagens.length);
  const imagensEmbaralhadas = Array.from(indices).map(i => imagens[i]);
  const labelsEmbaralhados  = Array.from(indices).map(i => labels[i]);

  const xs = tf.stack(imagensEmbaralhadas);
  const ys = tf.oneHot(tf.tensor1d(labelsEmbaralhados, 'int32'), CLASSES.length);

  // Liberar tensores individuais da memória
  imagens.forEach(t => t.dispose());

  return { xs, ys };
}

// ─── Arquitetura do modelo ──────────────────────────────────────────────────

function criarModelo() {
  const modelo = tf.sequential();

  modelo.add(tf.layers.conv2d({
    inputShape: [IMG_SIZE, IMG_SIZE, 3],
    filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'
  }));
  modelo.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  modelo.add(tf.layers.conv2d({
    filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'
  }));
  modelo.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  modelo.add(tf.layers.conv2d({
    filters: 128, kernelSize: 3, activation: 'relu', padding: 'same'
  }));
  modelo.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  modelo.add(tf.layers.flatten());
  modelo.add(tf.layers.dropout({ rate: 0.3 }));
  modelo.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  modelo.add(tf.layers.dense({ units: CLASSES.length, activation: 'softmax' }));

  modelo.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return modelo;
}

// ─── Treinamento ────────────────────────────────────────────────────────────

async function treinar() {
  console.log('\n📂 Carregando dataset...');
  const { xs, ys } = await carregarDataset();
  console.log(`Total de imagens: ${xs.shape[0]}\n`);

  const modelo = criarModelo();
  modelo.summary();

  console.log('\n🧠 Iniciando treinamento...\n');

  await modelo.fit(xs, ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationSplit: 0.2,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const acc    = (logs.acc * 100).toFixed(1);
        const valAcc = (logs.val_acc * 100).toFixed(1);
        console.log(
          `Epoch ${epoch + 1}/${EPOCHS} — ` +
          `loss: ${logs.loss.toFixed(4)} — acc: ${acc}% — ` +
          `val_loss: ${logs.val_loss.toFixed(4)} — val_acc: ${valAcc}%`
        );
      }
    }
  });

  console.log('\n💾 Salvando modelo...');
  await modelo.save(SAIDA_MODELO);
  console.log('Modelo salvo em ./modelo-exportado/');
  console.log('Copie model.json e os arquivos .bin para a pasta modelo/ do projeto.');

  xs.dispose();
  ys.dispose();
}

treinar().catch(console.error);
```

#### A.3 — Executar e avaliar

```bash
node treinar.js
```

**O que observar:**
- `val_acc` subindo a cada epoch — é o número que importa
- Se `acc` sobe mas `val_acc` fica estagnada: overfitting → aumente o Dropout para 0.4
- Resultado esperado após 30 epochs: `val_acc` > 85%

**Após o treinamento:**

Copie os arquivos gerados para o projeto principal:

```bash
cp treinamento/modelo-exportado/model.json     projeto-ratas/modelo/
cp treinamento/modelo-exportado/*.bin          projeto-ratas/modelo/
```

---

### Opção B — Google Colab

**Quando usar:** você não quer instalar nada localmente, quer usar a GPU gratuita do
Google, ou prefere o ambiente Python/Keras (mais documentação disponível).

O Colab treina com Keras (Python) e depois converte o modelo para o formato
TensorFlow.js que o navegador entende.

#### B.1 — Preparar o dataset no Google Drive

1. Acesse [drive.google.com](https://drive.google.com) e crie esta estrutura:

```
Meu Drive/
  ratas-dataset/
    cinzenta/
    labradora/
    nininha/
    bibranca/
```

2. Faça upload de todas as fotos nas pastas correspondentes.

#### B.2 — Notebook de treinamento

Crie um novo notebook no Colab e adicione as células abaixo, uma por vez.

---

**Célula 1 — Montar o Drive e instalar conversor**

```python
from google.colab import drive
drive.mount('/content/drive')

# Instala o conversor de Keras → TensorFlow.js
!pip install tensorflowjs --quiet
```

---

**Célula 2 — Configuração e carregamento do dataset**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path

# ── Configuração ────────────────────────────────────────────────────────────
DATASET_DIR = '/content/drive/MyDrive/ratas-dataset'
CLASSES     = ['cinzenta', 'labradora', 'nininha', 'bibranca']
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30

# ── Carregamento com augmentation ───────────────────────────────────────────
# Data augmentation: gera variações das fotos para o modelo generalizar melhor.
# Inclui espelhar horizontalmente, rotacionar levemente e variar brilho.
# Só é aplicado no treino, não na validação.

datagen_treino = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,           # normaliza 0-255 → 0.0-1.0
    validation_split=0.2,
    horizontal_flip=True,     # espelha aleatoriamente
    rotation_range=15,        # rotaciona até 15 graus
    brightness_range=[0.8, 1.2],  # varia brilho em ±20%
    zoom_range=0.1
)

datagen_val = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

treino = datagen_treino.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validacao = datagen_val.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f'\nClasses: {treino.class_indices}')
print(f'Imagens de treino:    {treino.samples}')
print(f'Imagens de validação: {validacao.samples}')
```

---

**Célula 3 — Definir e treinar o modelo**

```python
# ── Arquitetura ─────────────────────────────────────────────────────────────
modelo = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                        input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(CLASSES), activation='softmax')
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
# EarlyStopping: para o treino se val_accuracy não melhorar por 5 epochs
# ModelCheckpoint: salva sempre a melhor versão do modelo
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        '/content/melhor_modelo.keras',
        monitor='val_accuracy', save_best_only=True
    )
]

# ── Treinamento ──────────────────────────────────────────────────────────────
historico = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=EPOCHS,
    callbacks=callbacks
)
```

---

**Célula 4 — Avaliar e converter para TensorFlow.js**

```python
# ── Avaliação final ──────────────────────────────────────────────────────────
perda, acuracia = modelo.evaluate(validacao)
print(f'\nAcurácia de validação: {acuracia * 100:.1f}%')

if acuracia < 0.80:
    print('⚠️  Acurácia abaixo de 80%. Considere coletar mais fotos ou usar Transfer Learning.')
else:
    print('✅ Acurácia satisfatória. Prossiga com a conversão.')

# ── Conversão para TensorFlow.js ─────────────────────────────────────────────
import subprocess

subprocess.run([
    'tensorflowjs_converter',
    '--input_format=keras',
    '/content/melhor_modelo.keras',
    '/content/modelo-tfjs'
], check=True)

print('\nArquivos gerados:')
for f in os.listdir('/content/modelo-tfjs'):
    print(f'  {f}')
```

---

**Célula 5 — Baixar os arquivos**

```python
import shutil

# Compacta a pasta do modelo em um zip
shutil.make_archive('/content/modelo-tfjs', 'zip', '/content/modelo-tfjs')

from google.colab import files
files.download('/content/modelo-tfjs.zip')
```

Descompacte o zip e copie `model.json` e os arquivos `.bin` para `projeto-ratas/modelo/`.

---

#### B.3 — Sobre Transfer Learning no Colab

Se após ajustes a acurácia não passar de 80%, use MobileNetV2 como base.
Troque a **Célula 3** por esta versão:

```python
# Carrega MobileNetV2 pré-treinada no ImageNet, sem as camadas de classificação
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False  # congela os pesos pré-treinados

modelo = keras.Sequential([
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(CLASSES), activation='softmax')
])

# Mesmo compile e fit da versão anterior
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Por que funciona melhor?** O MobileNetV2 já foi treinado em 14 milhões de imagens e
aprendeu a detectar bordas, texturas e formas genéricas. Você aproveita todo esse
conhecimento e só treina as camadas finais para suas 4 ratas.

---

## Fase 3 — Aplicação client-side

Com o `model.json` em mãos, agora você constrói os dois HTMLs que rodam tudo
no navegador.

### Passo 3.1 — Módulo IndexedDB (`db.js`)

Este arquivo é importado por `camera.html` e `dashboard.html`.
Centraliza toda interação com o banco local.

```js
// db.js — módulo de persistência com IndexedDB

const DB_NOME    = 'ratas-db';
const DB_VERSAO  = 1;
const STORE_NOME = 'capturas';

function abrirBanco() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NOME, DB_VERSAO);

    req.onupgradeneeded = (e) => {
      const db    = e.target.result;
      const store = db.createObjectStore(STORE_NOME, {
        keyPath: 'id', autoIncrement: true
      });
      store.createIndex('por_rata', 'rata_nome', { unique: false });
      store.createIndex('por_data', 'criado_em', { unique: false });
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

// Salva uma captura. imagemBlob é o arquivo JPEG da captura.
export async function inserirCaptura(rata_nome, confianca, imagemBlob) {
  const db  = await abrirBanco();
  const rec = {
    rata_nome,
    confianca,
    imagem: imagemBlob,          // Blob JPEG — IndexedDB aceita nativamente
    criado_em: new Date().toISOString()
  };

  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE_NOME, 'readwrite');
    const req = tx.objectStore(STORE_NOME).add(rec);
    req.onsuccess = () => resolve(req.result);   // retorna o id gerado
    req.onerror   = () => reject(req.error);
  });
}

// Lista todas as capturas, opcionalmente filtradas por rata.
// Retorna um array de objetos, do mais recente para o mais antigo.
export async function listarCapturas(filtro = null) {
  const db = await abrirBanco();

  return new Promise((resolve, reject) => {
    const tx     = db.transaction(STORE_NOME, 'readonly');
    const store  = tx.objectStore(STORE_NOME);
    const req    = filtro
      ? store.index('por_rata').getAll(filtro)
      : store.getAll();

    req.onsuccess = () => {
      const resultado = req.result.reverse(); // mais recente primeiro
      resolve(resultado);
    };
    req.onerror = () => reject(req.error);
  });
}

// Conta quantas vezes cada rata apareceu.
// Retorna um objeto: { cinzenta: 12, bibranca: 6, ... }
export async function contarPorRata() {
  const todas   = await listarCapturas();
  const contagem = { cinzenta: 0, labradora: 0, nininha: 0, bibranca: 0 };
  todas.forEach(c => {
    if (contagem[c.rata_nome] !== undefined) contagem[c.rata_nome]++;
  });
  return contagem;
}
```

---

### Passo 3.2 — Detector de movimento com inferência (`camera.html`)

Este arquivo roda no celular. Detecta movimento, classifica a imagem e persiste no IndexedDB.

**Estrutura:**

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Câmera — Ratas</title>
</head>
<body>

  <video id="video" autoplay playsinline muted
         style="width:100%; max-width:480px"></video>

  <canvas id="canvas-atual"    style="display:none"></canvas>
  <canvas id="canvas-anterior" style="display:none"></canvas>

  <div id="status">Iniciando câmera...</div>

  <!-- TensorFlow.js — versão browser (não precisa de Node) -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>

  <script type="module">
    import { inserirCaptura } from './db.js';

    // ── Configuração ────────────────────────────────────────────────────────
    const CLASSES         = ['cinzenta', 'labradora', 'nininha', 'bibranca'];
    const LIMIAR_MOVIMENTO = 0.15;   // 15% de pixels diferentes → movimento
    const INTERVALO_MS    = 500;     // verifica a cada 500ms
    const COOLDOWN_MS     = 5000;    // aguarda 5s entre capturas
    const IMG_SIZE        = 224;

    // ── Estado ──────────────────────────────────────────────────────────────
    let modelo          = null;
    let primeroFrame    = true;
    let ultimaCaptura   = 0;

    const video         = document.getElementById('video');
    const canvasAtual   = document.getElementById('canvas-atual');
    const canvasAnterior= document.getElementById('canvas-anterior');
    const ctxAtual      = canvasAtual.getContext('2d');
    const ctxAnterior   = canvasAnterior.getContext('2d');
    const statusEl      = document.getElementById('status');

    function setStatus(msg) { statusEl.textContent = msg; }

    // ── Carregar modelo ──────────────────────────────────────────────────────
    setStatus('Carregando modelo...');
    modelo = await tf.loadLayersModel('./modelo/model.json');
    setStatus('Modelo carregado. Aguardando câmera...');

    // ── Iniciar câmera ───────────────────────────────────────────────────────
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }, audio: false
    });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      canvasAtual.width    = video.videoWidth;
      canvasAtual.height   = video.videoHeight;
      canvasAnterior.width = video.videoWidth;
      canvasAnterior.height= video.videoHeight;
      setStatus('Monitorando...');
      setInterval(verificarMovimento, INTERVALO_MS);
    };

    // ── Detecção de movimento ────────────────────────────────────────────────
    function verificarMovimento() {
      if (primeroFrame) {
        ctxAnterior.drawImage(video, 0, 0);
        primeroFrame = false;
        return;
      }

      ctxAtual.drawImage(video, 0, 0);

      const dadosAtual    = ctxAtual.getImageData(0, 0, canvasAtual.width, canvasAtual.height).data;
      const dadosAnterior = ctxAnterior.getImageData(0, 0, canvasAnterior.width, canvasAnterior.height).data;

      let pixelsDiferentes = 0;
      const totalPixels    = dadosAtual.length / 4;

      for (let i = 0; i < dadosAtual.length; i += 4) {
        const dr = Math.abs(dadosAtual[i]     - dadosAnterior[i]);
        const dg = Math.abs(dadosAtual[i + 1] - dadosAnterior[i + 1]);
        const db = Math.abs(dadosAtual[i + 2] - dadosAnterior[i + 2]);
        if ((dr + dg + db) > 30) pixelsDiferentes++;
      }

      const pct = pixelsDiferentes / totalPixels;
      ctxAnterior.drawImage(canvasAtual, 0, 0);

      const agora = Date.now();
      if (pct > LIMIAR_MOVIMENTO && (agora - ultimaCaptura) > COOLDOWN_MS) {
        ultimaCaptura = agora;
        classificar();
      }
    }

    // ── Inferência ───────────────────────────────────────────────────────────
    async function classificar() {
      setStatus('Movimento detectado! Classificando...');

      // Cria canvas 224×224 para o modelo
      const canvasModelo = document.createElement('canvas');
      canvasModelo.width = canvasModelo.height = IMG_SIZE;
      canvasModelo.getContext('2d').drawImage(
        canvasAtual, 0, 0, canvasAtual.width, canvasAtual.height,
        0, 0, IMG_SIZE, IMG_SIZE
      );

      // Converte para tensor e normaliza
      const tensor = tf.tidy(() =>
        tf.browser.fromPixels(canvasModelo)
          .toFloat()
          .div(255)
          .expandDims(0)
      );

      const predicao    = modelo.predict(tensor);
      const probs       = await predicao.data();
      const indice      = probs.indexOf(Math.max(...probs));
      const rataNome    = CLASSES[indice];
      const confianca   = probs[indice];

      tensor.dispose();
      predicao.dispose();

      setStatus(`Rata: ${rataNome} (${Math.round(confianca * 100)}%)`);

      // Salva imagem como Blob JPEG no IndexedDB
      canvasAtual.toBlob(async (blob) => {
        await inserirCaptura(rataNome, confianca, blob);
        setStatus(`✅ Salvo: ${rataNome} — ${Math.round(confianca * 100)}% confiança`);
        setTimeout(() => setStatus('Monitorando...'), 2000);
      }, 'image/jpeg', 0.85);
    }
  </script>

</body>
</html>
```

---

### Passo 3.3 — Dashboard (`dashboard.html`)

Esta página lê o IndexedDB e exibe o histórico com foto, nome, horário e confiança.

**Estrutura:**

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard — Ratas</title>
  <style>
    body        { font-family: sans-serif; padding: 1rem; }
    #stats      { display: flex; gap: 1rem; margin-bottom: 1rem; }
    .stat-card  { background: #f0f0f0; padding: 1rem; border-radius: 8px; text-align: center; flex: 1; }
    #filtros    { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
    #filtros button        { padding: 0.4rem 0.8rem; border: 1px solid #999; border-radius: 4px; cursor: pointer; background: #fff; }
    #filtros button.ativo  { background: #333; color: #fff; border-color: #333; }
    #grid       { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
    .card       { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
    .card img   { width: 100%; aspect-ratio: 4/3; object-fit: cover; }
    .card-info  { padding: 0.6rem; font-size: 0.85rem; }
    .confianca-alta   { color: green; }
    .confianca-media  { color: orange; }
    .confianca-baixa  { color: red; }
    .vazio      { grid-column: 1/-1; text-align: center; color: #999; padding: 2rem; }
  </style>
</head>
<body>

  <h1>Identificador de Ratas</h1>

  <div id="stats">
    <div class="stat-card">Cinzenta<br><strong id="cnt-cinzenta">—</strong></div>
    <div class="stat-card">bibranca+Marrom<br><strong id="cnt-labradora">—</strong></div>
    <div class="stat-card">bibranca+Cinzenta<br><strong id="cnt-nininha">—</strong></div>
    <div class="stat-card">bibranca<br><strong id="cnt-bibranca">—</strong></div>
  </div>

  <div id="filtros">
    <button class="ativo" data-filtro="">Todas</button>
    <button data-filtro="cinzenta">Cinzenta</button>
    <button data-filtro="labradora">bibranca+Marrom</button>
    <button data-filtro="nininha">bibranca+Cinzenta</button>
    <button data-filtro="bibranca">bibranca</button>
  </div>

  <div id="grid"></div>

  <script type="module">
    import { listarCapturas, contarPorRata } from './db.js';

    const grid   = document.getElementById('grid');
    let filtroAtivo = '';

    // ── Estatísticas ─────────────────────────────────────────────────────────
    async function atualizarStats() {
      const contagem = await contarPorRata();
      ['cinzenta', 'labradora', 'nininha', 'bibranca'].forEach(r => {
        document.getElementById(`cnt-${r}`).textContent =
          `${contagem[r] || 0} vis.`;
      });
    }

    // ── Renderização dos cards ────────────────────────────────────────────────
    function classeConfianca(v) {
      if (v >= 0.85) return 'confianca-alta';
      if (v >= 0.65) return 'confianca-media';
      return 'confianca-baixa';
    }

    function formatarData(iso) {
      return new Date(iso).toLocaleString('pt-BR');
    }

    function criarCard(captura) {
      const url = URL.createObjectURL(captura.imagem); // Blob → URL temporária
      const pct = Math.round(captura.confianca * 100);
      const cls = classeConfianca(captura.confianca);

      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `
        <img src="${url}" alt="${captura.rata_nome}">
        <div class="card-info">
          <strong>${captura.rata_nome.replace('_', '+')}</strong><br>
          ${formatarData(captura.criado_em)}<br>
          <span class="${cls}">Confiança: ${pct}%</span>
        </div>
      `;
      return div;
    }

    async function renderizar() {
      const capturas = await listarCapturas(filtroAtivo || null);
      grid.innerHTML = '';

      if (capturas.length === 0) {
        grid.innerHTML = '<p class="vazio">Nenhuma captura encontrada.</p>';
        return;
      }

      capturas.forEach(c => grid.appendChild(criarCard(c)));
    }

    // ── Filtros ──────────────────────────────────────────────────────────────
    document.querySelectorAll('#filtros button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#filtros button').forEach(b => b.classList.remove('ativo'));
        btn.classList.add('ativo');
        filtroAtivo = btn.dataset.filtro;
        renderizar();
      });
    });

    // ── Inicialização e polling ──────────────────────────────────────────────
    atualizarStats();
    renderizar();
    setInterval(() => { atualizarStats(); renderizar(); }, 30000);
  </script>

</body>
</html>
```

---

## Fase 4 — Hospedar e acessar pelo celular

Diferente do projeto original, você não precisa de um servidor Node rodando em produção.
Qualquer servidor de arquivos estáticos funciona.

### Opção 1 — Rede local (mais simples para testes)

```bash
# Na pasta projeto-ratas/
npx serve .
# Acesse no celular: http://SEU_IP:3000/camera.html
```

Descubra seu IP: `hostname -I` (Linux/Mac) ou `ipconfig` (Windows).

**Atenção:** `getUserMedia` exige HTTPS ou `localhost`. Na rede local via IP,
alguns navegadores bloqueiam o acesso à câmera. Se isso ocorrer:

```bash
# Alternativa com HTTPS local
npx serve . --ssl-cert ./cert.pem --ssl-key ./key.pem
# Gere os certificados com: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes
```

### Opção 2 — GitHub Pages (acesso de qualquer lugar)

1. Crie um repositório público no GitHub
2. Faça push de todos os arquivos (incluindo a pasta `modelo/`)
3. Vá em *Settings → Pages → Deploy from branch → main*
4. Seu projeto estará em `https://seu-usuario.github.io/projeto-ratas/`

**HTTPS já vem incluído** — o celular vai conseguir acessar a câmera sem configuração extra.

### Opção 3 — Netlify (deploy em 1 minuto)

1. Arraste a pasta `projeto-ratas/` para [app.netlify.com/drop](https://app.netlify.com/drop)
2. Netlify gera uma URL HTTPS automaticamente

---

## Troubleshooting

| Sintoma | Causa | Solução |
|---|---|---|
| `model.json` não carrega | Caminho errado | Confirme que o arquivo está em `./modelo/model.json` |
| Câmera bloqueada no celular | HTTP sem HTTPS | Use GitHub Pages ou gere certificado local |
| `Cannot use import` no console | Faltou `type="module"` | Confirme `<script type="module">` nos HTMLs |
| IndexedDB vazio no dashboard | Dispositivos diferentes | Dashboard e câmera precisam rodar no mesmo navegador |
| Inferência muito lenta | Sem WebGL | Verifique se o navegador suporta WebGL (`about:gpu` no Chrome) |
| Acurácia baixa em produção | Dataset diferente do ambiente real | Colete fotos no mesmo local/iluminação do uso real |

---

## Resumo das diferenças entre as opções de treinamento

| | Opção A (Node.js local) | Opção B (Google Colab) |
|---|---|---|
| Instalação | Node + npm | Nenhuma |
| GPU | Depende do hardware local | GPU gratuita no Colab |
| Linguagem | JavaScript | Python |
| Data augmentation | Manual (adicionar ao script) | Incluída no `ImageDataGenerator` |
| Transfer Learning | Requer reescrita maior | Troca de ~5 linhas |
| Iteração | Rápida (edita localmente) | Depende de upload do dataset |
| Melhor para | Quem já usa Node e quer tudo em JS | Quem quer GPU sem instalar nada |

---

## Próximos passos

- **Acurácia baixa?** → Tente Transfer Learning (Opção B, Célula 3 alternativa)
- **Histórico em múltiplos dispositivos?** → Adicione um backend mínimo só para sincronizar o IndexedDB, ou use Firebase Firestore (serverless)
- **Notificações?** → A Web Push API permite notificações mesmo com o navegador fechado
- **Uso noturno?** → Combine com um LED infravermelho; o modelo pode precisar de retreino com fotos no escuro
