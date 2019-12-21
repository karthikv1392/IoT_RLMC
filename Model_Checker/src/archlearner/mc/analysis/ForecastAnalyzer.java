package archlearner.mc.analysis;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.Collections;

import archlearner.mc.configuration.*;

public class ForecastAnalyzer {
	
	
	public static ArrayList<Long> extractTimeline(Configuration c) {
		ArrayList<Long> tl = new ArrayList<Long>();
		tl.add(0L);
		for (HashMap.Entry<String,ConfigurationNode> e: c.getM_nodes().entrySet()) {
			if (Objects.equals(e.getValue().getM_type(),ConfigurationNode.NodeType.SENSOR)) {
				LinkedList<SensorForecastPeriod> forecast = e.getValue().getM_forecast();
				for (int i=0;i<forecast.size();i++) {
					tl.add(forecast.get(i).getM_end());
					//System.out.println("-"+forecast.get(i).getM_end());
				}
			}
		}
		Collections.sort(tl);
		tl= (ArrayList<Long>) tl.stream().distinct().collect(Collectors.toList());
		//System.out.println(m_tl.toString());
		return tl;
	}
	
	public static HashMap<String, ConfigurationNode.Mode> extractModeSnapshot(Configuration c, Long time) {
		HashMap<String, ConfigurationNode.Mode> modes = new HashMap<String, ConfigurationNode.Mode>();
		
		for (HashMap.Entry<String,ConfigurationNode> e: c.getM_nodes().entrySet()) {
			if (Objects.equals(e.getValue().getM_type(),ConfigurationNode.NodeType.SENSOR)) {
				LinkedList<SensorForecastPeriod> forecast = e.getValue().getM_forecast();
				for (int i=0;i<forecast.size();i++) {
					if (!modes.containsKey(e.getValue().getM_id()))
						modes.put(e.getValue().getM_id(), forecast.get(i).getM_mode());
					if ((forecast.get(i).getM_start()<=time) && (forecast.get(i).getM_end()>=time))
						modes.put(e.getValue().getM_id(), forecast.get(i).getM_mode());
				}
			}
		}
		return modes;
	}
	
    public static void main(String[] args) {
    	
    	String filename = "models/archlearner_spark_output_co.json";
    	Configuration c = ConfigurationParser.readConfiguration(filename);
    	System.out.println(c.toString());
    	ArrayList<Long> tl = extractTimeline(c);
    	for (int i=0; i<tl.size();i++) {
        	System.out.println("Time: "+tl.get(i)+" : "+extractModeSnapshot(c,tl.get(i)));
    	}
    }

}
