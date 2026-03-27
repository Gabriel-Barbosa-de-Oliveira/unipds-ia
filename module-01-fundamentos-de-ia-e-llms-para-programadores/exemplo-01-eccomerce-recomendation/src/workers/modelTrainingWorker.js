import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

// Normalize continuous values (price, age) to 0-1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(catalog, users) {
    const ages = users.map(u => u.age);
    const price = catalog.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...price);
    const maxPrice = Math.max(...price);

    const colors = [...new Set(catalog.map(p => p.color))]; //new Set para remover duplicados
    const categories = [...new Set(catalog.map(p => p.category))];


    const colorIndex = Object.entries(
        colors.map((color, index) => {
            return [color, index];
        })
    );

    const categoriesIndex = Object.entries(
        categories.map((category, index) => {
            return [category, index];
        })
    );

    //Computar a media de idade dos compradores por produto 
    //(ajuda a entender quais produtos são mais populares entre diferentes faixas etárias)

    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age; // Soma de todas as idades 
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1; // Contagem de quantas vezes o produto foi comprado
        });
    })

    const productAvgAgeNorm = Object.fromEntries(
        catalog.map(product => {
            const avg =
                ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name]
                    : midAge; // Se o produto não foi comprado, usa a média geral
            return [product.name, normalize(avg, minAge, maxAge)];
        })
    )

    return {
        catalog,
        users,
        colorIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        //(price + age) + colors + categories
        dimensions: 2 + colors.length + categories.length, 
    }

}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('/data/products.json')).json();
    debugger
    const context = makeContext(catalog, users);

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
