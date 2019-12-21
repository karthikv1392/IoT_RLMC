package archlearner.mc.analysis;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Objects;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;

import archlearner.mc.codegeneration.PrismGenerator;
import archlearner.mc.configuration.*;
import archlearner.mc.configuration.ConfigurationNode.NodeType;
import parser.ast.ModulesFile;
import parser.ast.PropertiesFile;
import prism.Prism;
import prism.PrismDevNullLog;
import prism.PrismException;
import prism.PrismLog;
import prism.Result;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;



public class Analyzer
{

	private boolean m_debug = true;
	ModulesFile m_modulesfile;
	Prism m_prism;
	PrismGenerator m_pg;
	private boolean m_quantify_individual_energy = false;
	
	private static final String PCTL_TRAFFIC = "R{\"traffic\"}=?[C<=#]";
	private static final String PCTL_ENERGY = "R{\"energy_*\"}=? [C<=#]";
	private static final String PCTL_TOTALENERGY = "R{\"energy\"}=? [C<=#]";
	
	private static Double m_total_energy_result;
	private static Double m_total_traffic_result;
	private static LinkedList<TrafficPeriod> m_traffic_result;
		
	private String modelCheck(String property) {
		try{
			PropertiesFile propertiesFile = m_prism.parsePropertiesString(m_modulesfile, property);
	
			if (m_debug)
				System.out.println(propertiesFile.getPropertyObject(0));
			Result result = m_prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
			String res = result.getResult().toString();
			if (m_debug)
				System.out.println(res);
			return res;
		} catch (PrismException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		}
		return "";
	}

	public Double quantifyTraffic(int timeBound) {
		return Double.parseDouble(modelCheck(PCTL_TRAFFIC.replace("#", Integer.toString(timeBound))));
	}
	
	public Double quantifyTotalEnergy(int timeBound) {
		return Double.parseDouble(modelCheck(PCTL_TOTALENERGY.replace("#", Integer.toString(timeBound))));
	}
	
	public HashMap<String,Double> quantifyEnergy (int timeBound) {
		HashMap<String,Double> result = new HashMap<String,Double>();
		Configuration c = m_pg.getConfiguration();
		for (HashMap.Entry<String,ConfigurationNode> e: c.getM_nodes().entrySet()) {
			ConfigurationNode n = e.getValue();
			if (Objects.equals(n.getM_type(), NodeType.SENSOR)) {
				Double res = Double.parseDouble(modelCheck(PCTL_ENERGY.replace("*", n.getM_id()).replace("#", Integer.toString(timeBound))));
				result.put(e.getKey(), res);
			}
		}
		return result;
	}
	
	public Double analyze() {
    	Double traffic = 0.0;
    	Double traffic_partial = 0.0;
    	Double totalEnergy = 0.0;
    	HashMap <String,Double> energy = new HashMap<String,Double>();
    	m_traffic_result = new LinkedList<TrafficPeriod>();
		ArrayList<Long> tl = ForecastAnalyzer.extractTimeline(m_pg.getConfiguration());
		try {
	    	for (int i=0; i<tl.size()-1;i++) {
	    		m_modulesfile = m_prism.parseModelString(m_pg.getModelCode(ForecastAnalyzer.extractModeSnapshot(m_pg.getConfiguration(), tl.get(i))));
				m_prism.loadPRISMModel(m_modulesfile);
				int timeBound = ((Long)(tl.get(i+1)-tl.get(i))).intValue();
				traffic_partial = quantifyTraffic(timeBound);
				traffic += traffic_partial;
				m_traffic_result.add(new TrafficPeriod(tl.get(i).doubleValue(),tl.get(i+1).doubleValue(),traffic_partial));
				totalEnergy+= quantifyTotalEnergy(timeBound);
				if (m_quantify_individual_energy) {
					HashMap <String,Double> auxEnergy = quantifyEnergy(timeBound);
					for (HashMap.Entry<String,Double> e: auxEnergy.entrySet()) {
						if (!energy.containsKey(e.getKey()))
							energy.put(e.getKey(), auxEnergy.get(e.getKey()));
						else {
							Double currentEnergy = energy.get(e.getKey());
							energy.put(e.getKey(), currentEnergy+auxEnergy.get(e.getKey()));
						}
					}
				}
	    	}
		} catch (PrismException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		}

		m_total_traffic_result = traffic;
		if (m_debug)
			System.out.println("\nTraffic: " + traffic);
		
		m_total_energy_result = totalEnergy;
		if (m_debug)
			System.out.println("\nEnergy: " + totalEnergy);
		
		if (m_quantify_individual_energy)
			System.out.println("\nEnergy: " + energy );
		return 0.0;
	}
	
	public void exportToJSON(String filename) {
		JSONObject res = new JSONObject();
		res.put("energy_total", m_total_energy_result);
		res.put("traffic_total", m_total_traffic_result);
		
		JSONArray traffic = new JSONArray();
		for (int i=0; i<m_traffic_result.size();i++) {
			JSONObject trafficPeriod = new JSONObject();
			trafficPeriod.put("start", m_traffic_result.get(i).getM_start()); 
			trafficPeriod.put("end", m_traffic_result.get(i).getM_end());
			trafficPeriod.put("volume", m_traffic_result.get(i).getM_volume()); 
			traffic.add(trafficPeriod);
		}
		
		res.put("traffic_periods", traffic);
		
		try (FileWriter file = new FileWriter(filename)) {
			 
            file.write(res.toJSONString());
            file.flush();
 
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	
	public void run(String filename)
	{
		m_pg = new PrismGenerator(filename);
		
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			//PrismLog mainLog = new PrismFileLog("stdout");

			// Initialise PRISM engine 
			m_prism = new Prism(mainLog);
			m_prism.initialise();

			// Parse and load a PRISM model from a file
			//m_modulesfile = m_prism.parseModelString(m_pg.getModelCode(ForecastAnalyzer.extractModeSnapshot(m_pg.getConfiguration(), 100L)));
			//m_prism.loadPRISMModel(m_modulesfile);
			
			//System.out.println(quantifyTraffic(60));
			//System.out.println (quantifyEnergy(60));
			
			analyze();
			
			// Close down PRISM
			m_prism.closeDown();

		} catch (PrismException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		}
	}

	public static void main(String[] args)
	{
		
    	String input = args[0];
    	String output = args[1];

//    	String input = "models/archlearner_spark_output_sc.json";
//    	String output ="output/results.json";

    	
    	
    	Analyzer a = new Analyzer();
		a.run(input);
		a.exportToJSON(output);
	}

}