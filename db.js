// db.js — módulo de persistência com IndexedDB

const DB_NOME = 'ratas-db';
const DB_VERSAO = 1;
const STORE_NOME = 'capturas';

// Classes reconhecidas pelo modelo (3 ratas).
export const CLASSES = ['cinzenta', 'labradora', 'bibranca'];

function abrirBanco() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NOME, DB_VERSAO);

    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      const store = db.createObjectStore(STORE_NOME, {
        keyPath: 'id',
        autoIncrement: true,
      });
      store.createIndex('por_rata', 'rata_nome', { unique: false });
      store.createIndex('por_data', 'criado_em', { unique: false });
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// Salva uma captura. imagemBlob é o arquivo JPEG da captura.
export async function inserirCaptura(rata_nome, confianca, imagemBlob) {
  const db = await abrirBanco();
  const rec = {
    rata_nome,
    confianca,
    imagem: imagemBlob, // Blob JPEG — IndexedDB aceita nativamente
    criado_em: new Date().toISOString(),
  };

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NOME, 'readwrite');
    const req = tx.objectStore(STORE_NOME).add(rec);
    req.onsuccess = () => resolve(req.result); // retorna o id gerado
    req.onerror = () => reject(req.error);
  });
}

// Lista todas as capturas, opcionalmente filtradas por rata.
// Retorna um array de objetos, do mais recente para o mais antigo.
export async function listarCapturas(filtro = null) {
  const db = await abrirBanco();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NOME, 'readonly');
    const store = tx.objectStore(STORE_NOME);
    const req = filtro
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
// Retorna um objeto: { cinzenta: 12, labradora: 6, bibranca: 3 }
export async function contarPorRata() {
  const todas = await listarCapturas();
  const contagem = Object.fromEntries(CLASSES.map((c) => [c, 0]));
  todas.forEach((c) => {
    if (contagem[c.rata_nome] !== undefined) contagem[c.rata_nome]++;
  });
  return contagem;
}
