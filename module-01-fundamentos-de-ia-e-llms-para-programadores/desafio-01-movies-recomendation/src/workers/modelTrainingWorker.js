import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    rating: 0.4,       // avaliação - mais importante para recomendação
    genre: 0.35,       // genero - influencia preferências de estilo
    language: 0.3,     // linguagem - importante para recomendação
    director: 0.25,    // diretor - influencia estilo do filme
    year: 0.1,         // temporal
};
// Normalize continuous values (price, age) to 0-1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

/**
 * ====================================================================
 * 🏗️ CONSTRUIR CONTEXTO - Onde toda a mágica começa!
 * ====================================================================
 * 
 * Este método:
 * 1. Extrai valores únicos (gêneros, diretores, idiomas)
 * 2. Cria índices para codificação one-hot
 * 3. Calcula idade média dos assistidores por filme
 * 4. Retorna contexto pronto para encoding
 */
function makeContext(movies, users) {
    console.log('🎬 Construindo contexto de filmes...');
    console.log(`   📽️ Filmes: ${movies.length}`);
    console.log(`   👥 Usuários: ${users.length}`);

    // ====================================================================
    // 1️⃣ EXTRAIR RANGES DE VALORES CONTÍNUOS
    // ====================================================================

    // Idades dos usuários (para normalizar "idade média de quem assistiu")
    const ages = users.map(u => u.age);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    // Ratings dos filmes (0-10)
    const ratings = movies.map(m => m.rating);
    const minRating = Math.min(...ratings);
    const maxRating = Math.max(...ratings);

    // Anos de lançamento (1998-2021)
    const years = movies.map(m => m.year);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);

    // Durações dos filmes (em minutos: 98-169)
    const durations = movies.map(m => m.duration);
    const minDuration = Math.min(...durations);
    const maxDuration = Math.max(...durations);

    console.log(`   📊 Ranges - Age: [${minAge}-${maxAge}], Rating: [${minRating}-${maxRating}], Year: [${minYear}-${maxYear}], Duration: [${minDuration}-${maxDuration}]`);

    // ====================================================================
    // 2️⃣ EXTRAIR CATEGORIAS ÚNICAS & CRIAR ÍNDICES
    // ====================================================================

    // Gêneros únicos (ficção científica, ação, drama, crime, documentário)
    const genres = [...new Set(movies.map(m => m.genre))];

    // Diretores únicos (Christopher Nolan, Walter Salles, etc)
    const directors = [...new Set(movies.map(m => m.director))];

    // Idiomas únicos (english, português)
    const languages = [...new Set(movies.map(m => m.language))];

    console.log(`   🎭 Gêneros (${genres.length}):`, genres);
    console.log(`   🎥 Diretores (${directors.length}):`, directors.slice(0, 5), '...');
    console.log(`   🗣️ Idiomas (${languages.length}):`, languages);

    // ====================================================================
    // 3️⃣ CRIAR DICIONÁRIOS: MAPEAMENTO DE ÍNDICES
    // ====================================================================
    // Exemplo: genresIndex = { 'ficção científica': 0, 'ação': 1, 'drama': 2, ... }
    // Isso permite fazer one-hot encoding depois!

    const genresIndex = Object.fromEntries(
        genres.map((genre, index) => [genre, index])
    );

    const directorsIndex = Object.fromEntries(
        directors.map((director, index) => [director, index])
    );

    const languagesIndex = Object.fromEntries(
        languages.map((language, index) => [language, index])
    );

    // ====================================================================
    // 4️⃣ CALCULAR IDADE MÉDIA DE QUEM ASSISTIU CADA FILME
    // ====================================================================
    // 
    // Por quê? Ajuda o modelo a entender que:
    //   - Se um filme foi muito assistido por jovens (idade 22-25)
    //   - Um novo usuário jovem terá preferência maior por esse filme
    //
    // Exemplo: "Dune" (2021) pode ser mais assistido por jovens (média 25 anos)
    //          "Central do Brasil" (1998) pode ser mais assistido por adultos (média 30 anos)
    //

    const midAge = (minAge + maxAge) / 2; // média geral de idade
    const ageSums = {};      // soma de idades que assistiram cada filme
    const ageCounts = {};    // contagem de quantas pessoas assistiram

    // Iterar por cada usuário e seus filmes assistidos
    users.forEach(user => {
        user.movieWatches.forEach(movie => {
            ageSums[movie.name] = (ageSums[movie.name] || 0) + user.age;
            ageCounts[movie.name] = (ageCounts[movie.name] || 0) + 1;
        });
    });

    // Calcular a idade NORMALIZADA média para cada filme
    const movieAvgAgeNorm = Object.fromEntries(
        movies.map(movie => {
            // Se ninguém assistiu esse filme ainda, usar idade média geral
            const avg = ageCounts[movie.name]
                ? ageSums[movie.name] / ageCounts[movie.name]
                : midAge;

            // Normalizar para 0-1
            const normalized = normalize(avg, minAge, maxAge);

            return [movie.name, normalized];
        })
    );

    console.log(`   ✅ Idade média normalizada calculada para ${Object.keys(movieAvgAgeNorm).length} filmes`);

    // ====================================================================
    // 5️⃣ CALCULAR DIMENSIONALIDADE DO VETOR
    // ====================================================================
    // 
    // Dimensão total = quantos números cada filme será representado
    // 
    // Breakdown:
    //   - Rating (normalizado 0-1): 1 valor
    //   - Year (normalizado 0-1): 1 valor
    //   - Duration (normalizado 0-1): 1 valor
    //   - Genre (one-hot): N_GENRES valores (5 gêneros = 5 valores)
    //   - Director (one-hot): N_DIRECTORS valores (varia)
    //   - Language (one-hot): N_LANGUAGES valores (2 idiomas = 2 valores)
    //
    // Total = 3 + 5 + N_DIRECTORS + 2
    //

    const dimensions = 3 + genres.length + directors.length + languages.length;

    console.log(`
    ✅ CONTEXTO CONSTRUÍDO COM SUCESSO!
    ├─ Dimensões do vetor de filme: ${dimensions}
    │  ├─ Valores contínuos: 3 (rating, year, duration)
    │  ├─ Gêneros (one-hot): ${genres.length}
    │  ├─ Diretores (one-hot): ${directors.length}
    │  └─ Idiomas (one-hot): ${languages.length}
    ├─ Dimensões total (user+movie): ${dimensions * 2}
    └─ Dados: ${movies.length} filmes × ${users.length} usuários
    `);

    // ====================================================================
    // 6️⃣ RETORNAR CONTEXTO COMPLETO
    // ====================================================================
    return {
        // 📊 Dados brutos
        movies,                    // Array de filmes original
        users,                     // Array de usuários original

        // 📈 Índices para one-hot encoding
        genresIndex,              // { 'ficção científica': 0, 'ação': 1, ... }
        directorsIndex,           // { 'Christopher Nolan': 0, 'Walter Salles': 1, ... }
        languagesIndex,           // { 'english': 0, 'português': 1 }

        // 🔢 Ranges para normalização
        minAge,
        maxAge,
        minRating,
        maxRating,
        minYear,
        maxYear,
        minDuration,
        maxDuration,
        // 📊 Metadados
        movieAvgAgeNorm,          // { 'Inception': 0.45, 'Cidade de Deus': 0.8, ... }
        // 📐 Contagens
        numGenres: genres.length,
        numDirectors: directors.length,
        numLanguages: languages.length,

        // 🎯 Dimensionalidade (importante para rede neural!)
        dimensions: dimensions
    };
}


const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

function encodeMovie(movie, context) {
    const rating = tf.tensor1d([
        normalize(movie.rating, context.minRating, context.maxRating) * WEIGHTS.rating
    ]);

    const year = tf.tensor1d([
        normalize(movie.year, context.minYear, context.maxYear) * WEIGHTS.year
    ]);

    const duration = tf.tensor1d([
        normalize(movie.duration, context.minDuration, context.maxDuration) * 0.05
    ]);

    const genre = oneHotWeighted(
        context.genresIndex[movie.genre],
        context.numGenres,
        WEIGHTS.genre
    );

    const director = oneHotWeighted(
        context.directorsIndex[movie.director],
        context.numDirectors,
        WEIGHTS.director
    );

    const language = oneHotWeighted(
        context.languagesIndex[movie.language],
        context.numLanguages,
        WEIGHTS.language
    );

    return tf.concat1d([rating, year, duration, genre, director, language]);
}

function encodeUser(user, context) {

    // ====================================================================
    // CASO 1️⃣: USUÁRIO COM HISTÓRICO (assistiu filmes)
    // ====================================================================
    if (user.movieWatches && user.movieWatches.length > 0) {
        console.log(`👤 Codificando usuário "${user.name}" (${user.age} anos)`);
        console.log(`   📽️ Filmes assistidos: ${user.movieWatches.length}`);

        // ┌─────────────────────────────────────────────────────────┐
        // │ PASSO 1: Codificar CADA filme assistido                │
        // └─────────────────────────────────────────────────────────┘
        // 
        // Exemplo: usuário assistiu 3 filmes
        // filme1 → [0.35, 0.07, ..., 0.9]  (vetor 1)
        // filme2 → [0.25, 0.10, ..., 0.8]  (vetor 2)
        // filme3 → [0.15, 0.08, ..., 0.7]  (vetor 3)
        //
        const movieVectors = user.movieWatches.map(movie => {
            const encoded = encodeMovie(movie, context);
            console.log(`     ✅ "${movie.name}" codificado`);
            return encoded;
        });

        // ┌─────────────────────────────────────────────────────────┐
        // │ PASSO 2: EMPILHAR todos os vetores                     │
        // └─────────────────────────────────────────────────────────┘
        // 
        // shape: [3, 18] (3 filmes, 18 dimensões cada)
        //
        const stacked = tf.stack(movieVectors);

        // ┌─────────────────────────────────────────────────────────┐
        // │ PASSO 3: CALCULAR MÉDIA (mean)                         │
        // └─────────────────────────────────────────────────────────┘
        // 
        // Média coluna-por-coluna (axis=0)
        // [0.35, 0.07, ..., 0.9]
        // [0.25, 0.10, ..., 0.8]
        // [0.15, 0.08, ..., 0.7]
        // ───────────────────────────
        // [0.25, 0.08, ..., 0.8]  ← MÉDIA (perfil do usuário)
        //
        // shape: [18] (1D)
        //
        const meanVector = stacked.mean(0);

        // ┌─────────────────────────────────────────────────────────┐
        // │ PASSO 4: RESHAPE para [1, 18] (requerido pelo modelo)  │
        // └─────────────────────────────────────────────────────────┘
        //
        // O modelo sempre espera batch de exemplos, mesmo com 1 usuário
        // shape: [1, 18] (1 lote, 18 dimensões)
        //
        const reshaped = meanVector.reshape([1, context.dimensions]);

        console.log(`   ✅ Perfil do usuário: ${context.dimensions}D`);
        return reshaped;
    }

    // ====================================================================
    // CASO 2️⃣: USUÁRIO NOVO (sem histórico de filmes)
    // ====================================================================
    else {
        console.log(`👤 Codificando usuário NOVO "${user.name}" (${user.age} anos)`);
        console.log(`   📽️ Sem histórico de filmes`);

        // ┌─────────────────────────────────────────────────────────┐
        // │ Para novo usuário: usar APENAS a idade                │
        // │ (é o único dado que temos)                            │
        // └─────────────────────────────────────────────────────────┘
        //
        // Estratégia: preencher com zeros as partes de "gostos"
        //            e apenas ter a idade normalizada
        //
        // Estrutura do vetor:
        // [rating_zero, year_norm_idade, duration_zero, 
        //  genre_zeros, director_zeros, language_zeros]
        //
        // Isso faz o modelo entender: "não sabemos os gostos,
        // mas sabemos a idade deste usuário"

        // 1. Rating → ignorado (zero)
        const ratingZero = tf.zeros([1]);

        // 2. Year normalizado → usar idade do usuário
        //    (como proxy: usuários mais velhos preferem filmes mais antigos)
        const ageNormalized = tf.tensor1d([
            normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.year
        ]);

        // 3. Duration → ignorado (zero)
        const durationZero = tf.zeros([1]);

        // 4. Genre → todos zeros
        const genreZeros = tf.zeros([context.numGenres]);

        // 5. Director → todos zeros
        const directorZeros = tf.zeros([context.numDirectors]);

        // 6. Language → todos zeros
        const languageZeros = tf.zeros([context.numLanguages]);

        // ┌──────────────────────��──────────────────────────────────┐
        // │ Concatenar tudo e reshape                              │
        // └─────────────────────────────────────────────────────────┘
        //
        // Resultado: [1, 0.4, 0, 0,0,0,0,0, 0,0,..., 0,0]
        //            ^                  ^              ^
        //            rating_zero    genre_zeros   language_zeros
        //                   age_normalized
        //
        const userVector = tf.concat1d([
            ratingZero,           // rating não é relevante para novo usuário
            ageNormalized,        // idade normalizada (única pista)
            durationZero,         // duration não é relevante
            genreZeros,           // sem preferências de gênero (novo usuário)
            directorZeros,        // sem preferências de diretor
            languageZeros         // sem preferências de idioma
        ]);

        // Reshape para [1, dimensions]
        const reshaped = userVector.reshape([1, context.dimensions]);

        console.log(`   ✅ Perfil baseado APENAS em idade: ${context.dimensions}D`);
        return reshaped;
    }
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeMovie(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ?
                        1 :
                        0
                )
                // combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)

            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================

async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential();
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01), //
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('/data/products.json')).json();
    const context = makeContext(products, users);
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeMovie(product, context).dataSync() // Convert tensor to regular array for easier storage
        }
    });

    _globalCtx = context;

    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);


    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend({ user }) {
    if (!_model) return;
    const context = _globalCtx
    // 1️⃣ Converta o usuário fornecido no vetor de features codificadas
    //    (preço ignorado, idade normalizada, categorias ignoradas)
    //    Isso transforma as informações do usuário no mesmo formato numérico
    //    que foi usado para treinar o modelo.

    const userVector = encodeUser(user, context).dataSync();

    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
    //    com o vetor codificado do produto.
    //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).


    const inputs = context.movieVectors.map(({ vector }) =>
        [...userVector, ...vector]  // ← ÚNICO CHANGE
    );

    // 3️⃣ Converta todos esses pares (usuário, produto) em um único Tensor.
    //    Formato: [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(inputs)

    // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
    //    O resultado é uma pontuação para cada produto entre 0 e 1.
    //    Quanto maior, maior a probabilidade do usuário querer aquele produto.
    const predictions = _model.predict(tf.tensor2d(inputs));

    // 5️⃣ Extraia as pontuações para um array JS normal.
    const scores = predictions.dataSync()
    const recommendations = context.movieVectors.map((item, index) => ({
        ...item.meta,
        score: scores[index]
    }));

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)

    // 8️⃣ Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
