source .env

API_URL="https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SITE_URL="http://localhost:3000"
OPENROUTER_SITE_NAME="My Example"

NLP_MODEL="google/gemma-4-31b-it:free"

curl --silent -X POST $API_URL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "HTTP-Referer: $OPENROUTER_SITE_URL" \
  -H "X-Title: $OPENROUTER_SITE_NAME" \
  -d '{"model": '"'$NLP_MODEL'"', "messages": [{"role":"user","content":"Me conte uma curiosidade sobre llms"}], "temperature": 0.3, "max_tokens": 1000}' | jq


# {
#   "id": "gen-1782837829-BVEzwihNPBiR1x3upEJa",
#   "object": "chat.completion",
#   "created": 1782837829,
#   "model": "google/gemma-4-31b-it-20260402:free",
#   "provider": "OpenInference",
#   "system_fingerprint": "vllm-0.21.1rc1.dev262+g33d7cbe02-tp8-dd71347d",
#   "service_tier": null,
#   "choices": [
#     {
#       "index": 0,
#       "logprobs": null,
#       "finish_reason": "stop",
#       "native_finish_reason": "stop",
#       "message": {
#         "role": "assistant",
#         "content": "Uma curiosidade fascinante sobre os LLMs (Large Language Models) é que eles **não \"sabem\" fatos no sentido humano, mas sim \"preveem\" a próxima peça de informação.**\n\nAqui está o detalhe técnico e curioso por trás disso:\n\n### O conceito de \"Tokens\" e Probabilidade\nUm LLM não lê palavras inteiras como nós. Ele quebra o texto em **tokens** (que podem ser palavras inteiras, sílabas ou até letras individuais). \n\nQuando você faz uma pergunta, o modelo não está consultando um banco de dados de \"verdades\". Em vez disso, ele está fazendo um cálculo matemático gigantesco de probabilidade. Ele olha para todos os tokens que já vieram antes e pergunta: **\"Estatisticamente, qual é o próximo token mais provável de aparecer aqui?\"**\n\n**Exemplo simples:**\nSe você escrever *\"O céu está...\"*, o modelo não \"olha para fora\" para ver a cor do céu. Ele analisa bilhões de frases que leu durante o treinamento e percebe que, na imensa maioria das vezes, a palavra que segue essa sequência é *\"azul\"*.\n\n### A \"Alucinação\" explicada\nIsso explica por que os LLMs às vezes **alucinam** (inventam fatos com total confiança). \n\nComo o objetivo do modelo é a **probabilidade linguística** e não a **veracidade factual**, se ele encontrar um padrão estatístico que *pareça* correto gramaticalmente e logicamente, ele vai gerar aquela resposta, mesmo que a informação seja falsa. Para o modelo, a frase \"O presidente do Brasil em 1800 era o astronauta João\" pode parecer estruturalmente correta, e se ele não tiver pesos estatísticos fortes o suficiente para corrigir isso, ele pode simplesmente \"completar o padrão\".\n\n### Em resumo:\nUm LLM é, essencialmente, um **\"autocompletar extremamente sofisticado\"**. A \"inteligência\" que percebemos surge da escala: quando você treina esse sistema com quase toda a internet, a capacidade de prever o próximo token torna-se tão precisa que o modelo consegue simular raciocínio lógico, programar código e escrever poesias.",
#         "refusal": null,
#         "reasoning": null
#       }
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 21,
#     "completion_tokens": 472,
#     "total_tokens": 493,
#     "cost": 0,
#     "is_byok": false,
#     "prompt_tokens_details": {
#       "cached_tokens": 0,
#       "cache_write_tokens": 0,
#       "audio_tokens": 0,
#       "video_tokens": 0
#     },
#     "cost_details": {
#       "upstream_inference_cost": 0,
#       "upstream_inference_prompt_cost": 0,
#       "upstream_inference_completions_cost": 0
#     },
#     "completion_tokens_details": {
#       "reasoning_tokens": 0,
#       "image_tokens": 0,
#       "audio_tokens": 0
#     }
#   }
# }