package au.gov.nla.dnn.inference.api.socket;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.json.JSONObject;
import au.gov.nla.dnn.inference.InferenceResult;
import au.gov.nla.dnn.inference.InferenceService;
import au.gov.nla.dnn.inference.api.InferenceApiAdapter;

public class SocketInferenceApiAdapter implements InferenceApiAdapter
{
    private static final int BYTE_BUFFER_SIZE = 1024;
    
    private ServerSocket serverSocket;
    private ExecutorService listenService;
    private ExecutorService processingService;
    private long shutdownWaitMax;
    private boolean stopped;
    
    public void initialise(Properties properties, InferenceService service, Consumer<Exception> errorHandler) throws Exception
    {
        String hostName = properties.getProperty("host", "");
        int port = Integer.parseInt(properties.getProperty("port", "2901"));
        int backlog = Integer.parseInt(properties.getProperty("backlog", "100"));
        int threadPoolSize = Integer.parseInt(properties.getProperty("thread-pool", "10"));
        shutdownWaitMax = Long.parseLong(properties.getProperty("max-shutdown-delay", "10000"));
        
        listenService = Executors.newSingleThreadExecutor();
        processingService = threadPoolSize==0?Executors.newCachedThreadPool():Executors.newFixedThreadPool(threadPoolSize);
        serverSocket = new ServerSocket(port, backlog, InetAddress.getByName(hostName));
        
        listenService.execute(new Runnable(){
            public void run()
            {
                try
                {
                    while(!stopped)
                    {
                        Socket socket = serverSocket.accept();
                        
                        processingService.execute(new Runnable() {
                            public void run()
                            {
                                process(socket, service, errorHandler);
                            }
                        });
                    }
                }
                catch(Exception e)
                {
                    errorHandler.accept(e);
                }
            }
        });
    }
    
    private void process(Socket socket, InferenceService service, Consumer<Exception> errorHandler)
    {
        try
        {
            try(BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                    PrintWriter writer = new PrintWriter(socket.getOutputStream()))
            {
                JSONObject response = new JSONObject();
                
                try
                {
                    InferenceResult result = service.infer(reader.readLine(), getPayload(socket.getInputStream()));
                    response.put("success", true);
                    response.put("result", result.toJSON());
                }
                catch(Exception e)
                {
                    response.put("success", false);
                    response.put("exception", e.getClass().getName()+": "+e.getMessage());
                }
                
                writer.println(response.toString());
                writer.flush();
            }
            finally
            {
                socket.close();
            }
        }
        catch(Exception e)
        {
            errorHandler.accept(e);
        }
    }
    
    private byte[] getPayload(InputStream in) throws IOException
    {
        try(ByteArrayOutputStream buffer = new ByteArrayOutputStream())
        {
            int nRead;
            byte[] data = new byte[BYTE_BUFFER_SIZE];
        
            while((nRead=in.read(data, 0, data.length))!=-1)
            {
                buffer.write(data, 0, nRead);
            }
        
            return buffer.toByteArray();
        }
    }
    
    public void dispose() throws Exception
    {
        if(!stopped)
        {
            stopped = true;
            
            listenService.shutdown();
            listenService.awaitTermination(shutdownWaitMax, TimeUnit.MILLISECONDS);
            
            processingService.shutdown();
            processingService.awaitTermination(shutdownWaitMax, TimeUnit.MILLISECONDS);  
        }
    }
}