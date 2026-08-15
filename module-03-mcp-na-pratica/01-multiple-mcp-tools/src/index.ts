import { readFileSync } from 'fs';
import { createServer } from './server.ts';
import { closeMCPTools } from './services/mcpService.ts';

const app = await createServer();

await app.listen({ port: 3000, host: '0.0.0.0' });
console.log(`Server is running on http://0.0.0.0:3000`);

const salesData = readFileSync('./data/sales.csv', 'utf-8');
// const salesData = readFileSync('./data/sales-complete.csv', 'utf-8');

const question = `
Rank the top 5 most sold products:

${salesData}
`
// const question =`
// Here is a CSV file called sales.csv.
// What's the total revenue from this sales data?.

// ${salesData}
// `

app.inject({
  method: 'POST',
  url: '/chat',
  payload: {
    question,
  },
}).then(async response => {
  console.log('Response from /chat:', response.statusCode)
  console.log(response.body);
  await closeMCPTools();
  process.exit(0);
}).catch(async error => {
  console.error('Error testing /chat endpoint:', error);
  await closeMCPTools();
  process.exit(1);
});
