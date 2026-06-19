# Connecting with OpenAI Clients

Once you have your LangGraph OpenAI Serve API running, you can connect to it using any OpenAI-compatible client. This tutorial shows how to interact with your API using various clients and libraries.

## Python OpenAI Client

The most common way to connect to your API is using the official OpenAI Python client:

```python
from openai import OpenAI

# Initialize client with your custom base URL
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Replace with your API URL
    api_key="any-value"  # API key is not verified in the default setup
)

# Making a standard chat completion request
response = client.chat.completions.create(
    model="custom-input-output-context",
    messages=[
        {"role": "user", "content": "Show me the custom adapter."},
    ],
    user="demo-user",
)

# Accessing the response
print(response.choices[0].message.content)
```

## Streaming Responses

To use streaming with the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-value"
)

# Create a streaming completion
stream = client.chat.completions.create(
    model="simple-graph-with-history",
    messages=[
        {"role": "user", "content": "Write a short poem about graphs."}
    ],
    stream=True  # Enable streaming
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## JavaScript/TypeScript Client

For web applications, you can use the OpenAI JavaScript client:

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'any-value',
  dangerouslyAllowBrowser: true // For client-side use
});

async function getChatCompletion() {
  const completion = await openai.chat.completions.create({
    model: 'custom-input-output-context',
    messages: [
      { role: 'user', content: 'Show me the custom adapter.' }
    ],
    user: 'demo-user',
  });

  console.log(completion.choices[0].message.content);
}

getChatCompletion();
```

## Streaming with JavaScript

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'any-value',
  dangerouslyAllowBrowser: true
});

async function streamChatCompletion() {
  const stream = await openai.chat.completions.create({
    model: 'simple-graph-with-history',
    messages: [
      { role: 'user', content: 'Write a short poem about graphs.' }
    ],
    stream: true,
  });

  let responseText = '';
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || '';
    responseText += content;
    console.log(content); // Update UI with new content
  }
}

streamChatCompletion();
```

## Using with curl

You can also use curl to interact with your API directly:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-input-output-context",
    "messages": [
      {"role": "user", "content": "Show me the custom adapter."}
    ],
    "user": "demo-user"
  }'
```

## Using with Other OpenAI Client Libraries

Most OpenAI client libraries in different programming languages will work with your LangGraph OpenAI Serve API, as long as they allow you to set a custom base URL. Here's a general pattern:

1. Initialize the client with your custom base URL
2. Set any API key (as it's not verified by default)
3. Make API calls as you would with the official OpenAI API

## Available Endpoints

The following endpoints are available in your LangGraph OpenAI Serve API:

- `GET /v1/models` - List available models (your registered graphs)
- `POST /v1/chat/completions` - Create a chat completion
- `GET /v1/health` - Check the health status of the API when routers are bound with `prefix="/v1"`

## Best Practices

1. **Error Handling**: Always implement proper error handling in your client code
2. **Timeouts**: Set appropriate request timeouts, especially for complex graph workflows
3. **API Key**: For production, consider implementing authentication and using proper API keys
4. **Model Selection**: Make sure to use the correct model name (graph name) in your requests
5. **Streaming**: For longer responses or better user experience, use streaming when possible

## Next Steps

- Learn about [deploying with Docker](../how-to-guides/docker.md)
- Explore how to [add authentication](../how-to-guides/authentication.md) to your API
- See the [API reference](../reference.md) for detailed endpoint documentation
