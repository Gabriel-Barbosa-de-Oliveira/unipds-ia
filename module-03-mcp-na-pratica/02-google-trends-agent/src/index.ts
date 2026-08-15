import { createServer } from './server.ts';

const app = await createServer();

await app.listen({ port: 3000, host: '0.0.0.0' });
console.log(`Server is running on http://0.0.0.0:3000`);


app.inject({
  method: 'POST',
  url: '/chat',
  payload: { question: "O que você me recomendaria para lançar uma nova musica no youtube chamada letargia ?" },
}).then(response => {
  console.log('Response from /chat:', response.statusCode)
  console.log(response.body);
}).catch(error => {
  console.error('Error testing /chat endpoint:', error);
});

// curl \
// -X POST \
// -H 'Content-type: application/json' \
// --data '{"question": "upper"}' \
// localhost: 3000 / chat
