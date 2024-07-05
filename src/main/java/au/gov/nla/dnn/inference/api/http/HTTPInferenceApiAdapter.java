package au.gov.nla.dnn.inference.api.http;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

import org.json.JSONObject;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import au.gov.nla.dnn.inference.InferenceResult;
import au.gov.nla.dnn.inference.InferenceService;
import au.gov.nla.dnn.inference.api.InferenceApiAdapter;

@SuppressWarnings("restriction")
public class HTTPInferenceApiAdapter implements InferenceApiAdapter
{
    private static final int BYTE_BUFFER_SIZE = 1024;
    
    private HttpServer server;
    private int shutdownDelay;

    private boolean allowCors;
    private String corsAllowedHosts;
    private String corsAllowedHeaders;
    
    public void initialise(Properties properties, InferenceService service, Consumer<Exception> errorHandler) throws Exception
    {
        String hostName = properties.getProperty("host", "");
        int port = Integer.parseInt(properties.getProperty("port", "2901"));
        int backlog = Integer.parseInt(properties.getProperty("backlog", "100"));
        int threadPoolSize = Integer.parseInt(properties.getProperty("thread-pool", "10"));
        shutdownDelay = Integer.parseInt(properties.getProperty("max-shutdown-delay-seconds", "0"));
        allowCors = Boolean.parseBoolean(properties.getProperty("cors-allowed", "false"));
        corsAllowedHosts = properties.getProperty("cors-allowed-hosts", "");
        corsAllowedHeaders = properties.getProperty("cors-allowed-headers", "");
        
        ExecutorService executor = threadPoolSize == 0 ? Executors.newCachedThreadPool():Executors.newFixedThreadPool(threadPoolSize);

        server = HttpServer.create((hostName.isEmpty() ? new InetSocketAddress(port) : new InetSocketAddress(hostName, port)), backlog);
        server.setExecutor(executor);
        
        server.createContext("infer", new HttpHandler(){
            public void handle(HttpExchange exchange) throws IOException
            {
                try
                {
                    if(allowCors)
                    {
                        exchange.getResponseHeaders().put("Access-Control-Allow-Headers", Arrays.asList(corsAllowedHeaders));
                        exchange.getResponseHeaders().put("Access-Control-Allow-Origin", Arrays.asList(corsAllowedHosts));
                    }
                    if(exchange.getRequestMethod().equals("OPTIONS"))
                    {
                        if(allowCors)
                        {
                            exchange.sendResponseHeaders(200, 0);
                            exchange.close();
                            return;
                        }
                        else
                        {
                            exchange.sendResponseHeaders(400, 0);
                            exchange.close();
                            return;
                        }
                    }
                    
                    String modelId = exchange.getRequestHeaders().getFirst("dnn-model-id");
                    
                    if(modelId==null || modelId.isBlank())
                    {
                        exchange.sendResponseHeaders(400, 0);
                        exchange.close();
                        throw new IOException("Missing header: dnn-model-id");
                    }
                    
                    JSONObject response = new JSONObject();
                    
                    try
                    {
                        InferenceResult result = service.infer(modelId, getPayload(exchange));
                        response.put("success", true);
                        response.put("result", result.toJSON());
                    }
                    catch(Exception e)
                    {
                        response.put("success", false);
                        response.put("exception", e.getClass().getName()+": "+e.getMessage());
                    }
                    
                    byte[] bytes = response.toString().getBytes();
                    
                    exchange.sendResponseHeaders(200, bytes.length);
                    exchange.getResponseBody().write(bytes);
                    exchange.getResponseBody().flush();
                    exchange.close();
                    return;
                }
                catch(Exception e)
                {
                    exchange.sendResponseHeaders(500, 0);
                    exchange.close();
                    errorHandler.accept(e);
                    throw new IOException(e);
                }
            }
            
            private byte[] getPayload(HttpExchange exchange) throws IOException
            {
                try(ByteArrayOutputStream buffer = new ByteArrayOutputStream())
                {
                    int nRead;
                    byte[] data = new byte[BYTE_BUFFER_SIZE];

                    while((nRead=exchange.getRequestBody().read(data, 0, data.length))!=-1)
                    {
                        buffer.write(data, 0, nRead);
                    }

                    return buffer.toByteArray();
                }
            }
        });
        
        server.start();
    }
    
    public void dispose() throws Exception
    {
        server.stop(shutdownDelay<=0 ? Integer.MAX_VALUE:shutdownDelay);
    }
}