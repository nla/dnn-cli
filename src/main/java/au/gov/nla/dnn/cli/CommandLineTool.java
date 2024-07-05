package au.gov.nla.dnn.cli;

import java.io.File;
import java.nio.file.Files;

import org.json.JSONObject;

import au.gov.nla.dnn.inference.InferenceServer;
import au.gov.nla.dnn.training.TrainingExecution;

public class CommandLineTool
{
    public static void main(String[] args) throws Exception
    {
        if(args.length>0)
        {
            switch(args[0])
            {
                case "-inference-server":
                {
                    if(args.length==2)
                    {
                        InferenceServer server = new InferenceServer();
                        server.start(loadConfig(args[1]));
                        return;
                    }
                    
                    break;
                }
                case "-train":
                {
                    if(args.length==5)
                    {
                        TrainingExecution execution = new TrainingExecution();
                        execution.execute(loadConfig(args[1]), args[2], args[3], args[4]);
                    }
                    
                    break;
                }
            }
        }
        
        System.out.println("Usage: (either)");
        System.out.println("-train <training-config-file> <temp-directory> <model-dest-file> <eval-dest-file>");
        System.out.println("-inference-server <server-config-file>");
        System.exit(1);
    }
    
    private static JSONObject loadConfig(String path) throws Exception
    {
        StringBuilder b = new StringBuilder();
        
        for(String line: Files.readAllLines(new File(path).toPath()))
        {
            b.append(line).append("\n");
        }
        
        return new JSONObject(b.toString());
    }
}