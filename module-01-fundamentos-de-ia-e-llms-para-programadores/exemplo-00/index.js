import tf, { callbacks } from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {

	const model = tf.sequential();

	// Primeira camada da rede:
	// Entrada de 7 posições (idade normalizada, 3 cores, 3 localizações)

	// 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino 
	// quanto mais neuronios, mais complexidade a rede pode aprender
	// e consequentemente, mais processamento ela vai usar 

	// A ReLu age como um filtro:
	// É como se ela deixasse somente os dados interessantes seguirem viagem na rede
	// Se a informação chegou nesse neuronio é positva, passa para frente! 
	// Se for zero ou negativa, pode jogar fora, não vai servir para nada!

	model.add(tf.layers.dense({
		inputShape: [7], // 7 características de entrada
		units: 80,       // Número de neurônios na camada oculta
		activation: 'relu' // Função de ativação ReLU
	}));

	// Saida: 3 neuronios
	// Um para cada categoria (premium, medium, basic)

	//activation: softmax normaliza a saida em probabilidades

	model.add(tf.layers.dense({
		units: 3,        // 3 categorias de saída
		activation: 'softmax' // Função de ativação softmax para classificação
	}));

	// Compilando o model 
	// optimizer: Adam (Adaptive Moment Estimation)
	// é um treinador pessoal moderno para redes neurais 
	// ajusta os pesos de forma eficiente e inteligente 
	// aprender com historico de erros e acertos

	// Loss: categoricalCrossentropy (Entropia Cruzada Categórica)
	// Ele compara o que o modelo "acha" (os scores de cada categoria) 
	// com a resposta correta (o label one-hot encoded)
	// a categoria premium tem que ser 1, e as outras 0, por exemplo [1,0,0]

	// Metrics: accuracy (Acurácia)
	// Quanto mais distante da previsão do modelo da resposta correto 
	// Maior o erro (loss)
	// Exmeplo Classico: Classificação de imagens, recomedação, categorizaçao 
	// de usuario
	// Qualquer coisa em que a resposta certe é "apenas uma entre varias possiveis"

	model.compile({
		optimizer: 'adam',
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	});

	//Treinamento de modelo
	//Verbose: desabilita o log interno (e usa só callback)
	//epochs: número de vezes que vai rodar no dataset
	//shuffle: embaralha os dados, para evitar que o modelo fique viciado

	await model.fit(
		inputXs,
		outputYs,
		{
			verbose: 0, //Não ficar colocando logs de cada passo
			epochs: 100, //Número de vezes que o modelo vai passar por todo o dataset de treino
			shuffle: true, //Embaralha os dados a cada época para evitar que o modelo aprenda padrões de ordem
			// callbacks: {
			// 	onEpochEnd: (epoch, log) => console.log(
			// 		`Epoch: ${epoch}: loss = ${log.loss}`
			// 	)
			// }
		}
	);

	//Pelos logs é possivel observar que o modelo começa com um erro alto
	//e vai diminuindo a medida que ele aprende com os dados de treino.
	//Exemplo 
	// Epoch: 0: loss = 1.069222331047058
	// Epoch: 1: loss = 1.0531052350997925
	// Epoch: 2: loss = 1.0373338460922241
	// Epoch: 3: loss = 1.0217403173446655
	// Epoch: 4: loss = 1.0063159465789795
	// Epoch: 5: loss = 0.9912128448486328
	// Epoch: 6: loss = 0.9763743877410889
	// Epoch: 7: loss = 0.9616788625717163
	// Epoch: 8: loss = 0.9471167922019958
	// Epoch: 9: loss = 0.9326667785644531
	// Epoch: 10: loss = 0.9183310270309448
	return model;
}

async function predict(model, pessoa) {
	// Transformar o array js para o tensor (tfjs)
	const tfInput = tf.tensor2d(pessoa);
	// Faz a predição (output sera um vetor de 3 probabilidades)

	const pred = model.predict(tfInput);

	const predArray = await pred.array();
	return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
	[0.33, 1, 0, 0, 1, 0, 0], // Erick
	[0, 0, 1, 0, 0, 1, 0],    // Ana
	[1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
	[1, 0, 0], // premium - Erick
	[0, 1, 0], // medium - Ana
	[0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// inputXs.print();
// outputYs.print();


// Quanto mais dado melhor !
// Assim o algoritmo consegue entender melhor os padrões complexos 
// dos dados
const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: "zé", idade: 28, cor: "verde", localizacao: "Curitiba" }
// Normalizando a idade da nova pessoa usando o mesmo padrão do treino 
// Exemplo: idade_min = 25, idade_max = 40, então (28-25) / (40-25) = 0.2

const pessoaTensorNormalizado = [
	[
		0.2, // idade normalizada
		0,   // azul
		0,   // vermelho
		1,   // verde
		0,   // São Paulo
		1,   // Rio
		0    // Curitiba
	] 
]

const predictions = await predict(model, pessoaTensorNormalizado);
const results = 
	predictions
	.sort((a, b) => b.prob - a.prob)
	.map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
	.join('\n');

console.log(results)
  