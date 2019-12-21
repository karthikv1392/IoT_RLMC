package archlearner.mc.configuration;


import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import archlearner.mc.configuration.SensorForecastPeriod;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Objects;
import java.util.LinkedList;

public class ConfigurationParser {

	private static boolean m_debug;
	
	
	
	private static void addNodes(JSONArray a, Configuration c, ConfigurationNode.NodeType type) {
		for (Object node: a) {
				c.addNode(new ConfigurationNode(node.toString(), type));
			}
	}

	private static void addSensorNodes (JSONObject o, Configuration c) {
		for (Object key: o.keySet()) {
			String sk = (String) key;
        	JSONObject sensor = (JSONObject) o.get(sk);
        	ConfigurationNode.Mode mode = Objects.equals(sensor.get("execution_mode").toString(),"critical")? ConfigurationNode.Mode.CRITICAL: ConfigurationNode.Mode.NORMAL;
        	JSONObject sensorModes =  (JSONObject) sensor.get("modes");
        	LinkedList<SensorForecastPeriod> forecast = new LinkedList<SensorForecastPeriod>();
        	
        	JSONArray jsonForecast = (JSONArray) sensor.get("forecasted_modes");
        	for (Object period: jsonForecast) {
        		ConfigurationNode.Mode periodMode = Objects.equals(((JSONObject) period).get("mode").toString(),"critical")? ConfigurationNode.Mode.CRITICAL: ConfigurationNode.Mode.NORMAL;
        		Long periodStart;
         		try {
        			periodStart = (Long) ((JSONObject) period).get("start");
        		} catch (Exception e) {
        			Double ps = (Double) ((JSONObject) period).get("start");
        			periodStart = ps.longValue();
        		}
        		Long periodEnd;
        		try {
        			periodEnd = (Long) ((JSONObject) period).get("end");
        		} catch (Exception e) {
        			Double pe = (Double) ((JSONObject) period).get("end");
        			periodEnd = pe.longValue();
        		}
        		forecast.add(new SensorForecastPeriod(periodStart, periodEnd, periodMode));
        	}
        	
        	
        	Double normalFreq;
        	Double criticalFreq;
        	try {
        		normalFreq = (Double) sensorModes.get("normal_freq");
        	} catch (Exception e) {
        		normalFreq = (Double) ((Long)sensorModes.get("normal_freq")).doubleValue();
        	}
        	try {
        		criticalFreq = (Double) sensorModes.get("critical_freq");
        	} catch (Exception e) {
        		criticalFreq = (Double) ((Long)sensorModes.get("critical_freq")).doubleValue();
        	}
        	
        	c.addNode(new ConfigurationNode(sk,
        			ConfigurationNode.NodeType.SENSOR,
        			mode,
        			1.0/normalFreq,
        			1.0/criticalFreq,
        			(Double)sensor.get("remaining_energy"),
        			(Double) sensor.get("idle_energy"),
        			forecast
        			));
		}
	}
	
	public static Configuration readConfiguration(String filename) {
		Configuration c = new Configuration();
    	
        JSONParser parser = new JSONParser();

        try (Reader reader = new FileReader(filename)) {

            JSONObject jsonObject = (JSONObject) parser.parse(reader);
            if (m_debug)
            	System.out.println(jsonObject);
            
            String type = (String) jsonObject.get("pattern");
            
            if (Objects.equals(type, "co"))
            	c.setM_type(Configuration.PatternType.CO);
            if (Objects.equals(type, "su"))
        		c.setM_type(Configuration.PatternType.SU);
            if (Objects.equals(type, "sc"))
        		c.setM_type(Configuration.PatternType.SC);
            
            JSONObject sensors = (JSONObject) jsonObject.get("sensors");
            JSONArray controllers = (JSONArray) jsonObject.get("controllers");
            JSONArray databases = (JSONArray) jsonObject.get("databases");
            JSONArray displays = (JSONArray) jsonObject.get("displays");
            JSONArray compute = (JSONArray) jsonObject.get("compute");
            addSensorNodes(sensors, c);
            try {
            	addNodes(controllers, c, ConfigurationNode.NodeType.CONTROLLER);
            } catch (Exception e) {
            	if (m_debug)
            		System.out.println("Warning: Controller list empty (SU pattern?)");
            }
            addNodes(databases, c, ConfigurationNode.NodeType.DATABASE);
            addNodes(displays, c, ConfigurationNode.NodeType.DISPLAY);
            addNodes(compute, c, ConfigurationNode.NodeType.COMPUTE);            
            
            JSONObject edges = (JSONObject) jsonObject.get("connections");
            
            for (Object key: edges.keySet()) {
            	String sk = (String) key;
            	JSONObject edge = (JSONObject) edges.get(sk);
            	c.addEdge(new ConfigurationEdge(sk, 
            			(Double) edge.get("sent"), 
            			(Double) edge.get("received"), 
            			(Double) edge.get("processing")));
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }

        return c;
	}
	
    public static void main(String[] args) {
    	
    	String filename = "models/archlearner_spark_output_sc.json";
    	Configuration c = ConfigurationParser.readConfiguration(filename);
    	System.out.println(c.toString());
    	
    }
    
}