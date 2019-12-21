package archlearner.mc.codegeneration;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Objects;

import archlearner.mc.analysis.ForecastAnalyzer;
import archlearner.mc.configuration.Configuration;
import archlearner.mc.configuration.Configuration.NeighborMode;
import archlearner.mc.configuration.Configuration.PatternType;
import archlearner.mc.configuration.ConfigurationNode;
import archlearner.mc.configuration.ConfigurationNode.Mode;
import archlearner.mc.configuration.ConfigurationNode.NodeType;
import archlearner.mc.configuration.ConfigurationEdge;
import archlearner.mc.configuration.ConfigurationParser;
import archlearner.mc.configuration.SensorForecastPeriod;

public class PrismGenerator {

	Configuration m_configuration;
	
	public PrismGenerator(String filename) {
		m_configuration = ConfigurationParser.readConfiguration(filename);
	}
	
	public Configuration getConfiguration () {
		return m_configuration;
	}
	
	public String getModelCode() {
		return getModelCode(null);
	}
	
	public String getModelCode(HashMap<String,ConfigurationNode.Mode> sensorModes) {
		String res = "ctmc\n\n";
		
		for (HashMap.Entry<String,ConfigurationNode> e: m_configuration.getM_nodes().entrySet()) {
			ConfigurationNode.Mode mode = Objects.equals(null, sensorModes) ? Mode.NONE : sensorModes.get(e.getKey());
			res += getNodeCode(e.getValue(), mode);
		}
		
		
		res += getRewardsCode();
		return res;
	}
	
	public String getNodeCode(ConfigurationNode n, ConfigurationNode.Mode mode){
		
		String res ="";
		
		switch (n.getM_type()) {
		case SENSOR:
				res += getSensorNodeCode(n, mode);
			break;
		case DATABASE:
		case DISPLAY:
		case COMPUTE:
		case CONTROLLER:
				res += getControllerNodeCode(n);
		break;
			
		}
			
		return res;
	}
	
	public String getSensorNodeCode(ConfigurationNode n, ConfigurationNode.Mode mode) {

		//String initModeStr = Objects.equals(n.getM_mode(),Mode.CRITICAL)? "true":"false";
		String initModeStr = Objects.equals(mode,Mode.CRITICAL)? "true":"false";
		String res="formula "+n.getM_id()+"_rate="+n.getM_id()+"_critical?"+n.getM_rate_critical()+":"+n.getM_rate_normal()+";\n";
		res+="module "+n.getM_id()+"\n";
//		if (Objects.equals(m_configuration.getM_type(),PatternType.SU))
//			res += "\t"+n.getM_id()+"_forward: bool init false;\n";
		res+="\t"+n.getM_id()+"_critical: bool init "+initModeStr+";\n";
		res+="\t"+n.getM_id()+"_order: [0.."+Integer.toString(n.getM_forecast().size())+"] init 0;\n";
		LinkedList<ConfigurationEdge> edges = m_configuration.getEdgesToNeighbors(n.getM_id(), NeighborMode.FROM);
		
		for (int i=0; i<edges.size(); i++) {
			res += getSensorCommandCode (n, edges.get(i));
		}
		
//		for (int i=0; i<n.getM_forecast().size();i++) {
//			res += getSensorPeriodChangeCommandCode(n, i);
//		}
		
		if (Objects.equals(m_configuration.getM_type(),PatternType.SU))
			res += getExtraSUSensorCommands(n);
		
		res += "endmodule\n\n";
		return res;
	}
	
	public String getExtraSUSensorCommands(ConfigurationNode n) {
		String res = "";
		String dbId = m_configuration.getDatabaseId();
		LinkedList<ConfigurationEdge> edges = m_configuration.getEdgesToNeighbors(n.getM_id(), NeighborMode.TO);
		for (int i=0; i<edges.size(); i++) {
			res += getSensorExtraSUCommandCode (n, edges.get(i), dbId);
		}
		return res;
	}
	
	public String getSensorExtraSUCommandCode (ConfigurationNode n, ConfigurationEdge e, String databaseId) {
		String res = "\t ["+e.getM_id()+"] true -> 1 :  true;\n";
//		String res = "\t ["+e.getM_id()+"] !"+n.getM_id()+"_forward -> 1 : ("+n.getM_id()+"_forward'=true);\n";
//		res += "\t ["+n.getM_id()+"_"+databaseId+"] "+n.getM_id()+"_forward -> 9999 : ("+n.getM_id()+"_forward'=false);\n";
		return res;
	}
	
	
	public String getSensorCommandCode (ConfigurationNode n, ConfigurationEdge e) {
		String res = "\t ["+e.getM_id()+"] true -> "+n.getM_id()+"_rate: true;\n";
		return res;
	}

	public String getSensorPeriodChangeCommandCode (ConfigurationNode n, int order) {
		String id = n.getM_id();
		SensorForecastPeriod fcp = n.getM_forecast().get(order);
		String rateStr = Double.toString(1.0/((Long)fcp.getM_end()).doubleValue());
		String modeStr = Objects.equals(fcp.getM_mode(), Mode.CRITICAL) ? "true" : "false";
		String res = "\t [] ("+id+"_order="+Integer.toString(order) + ") -> " + rateStr + ": ("+id+"_critical'="+modeStr+
				") & ("+id+"_order'="+Integer.toString(order)+"+1);\n";
		return res;
	}
	
	public String getControllerNodeCode(ConfigurationNode n) {

		String res="module "+n.getM_id()+"\n";
		res+="\t "+n.getM_id()+"_received: bool init false;\n";
		LinkedList<ConfigurationEdge> edges = m_configuration.getEdgesToNeighbors(n.getM_id(),NeighborMode.TO);
		
		for (int i=0; i<edges.size(); i++) {
			ConfigurationEdge f = edges.get(i);
			res += "\t["+f.getM_id()+"] (!"+n.getM_id()+"_received) -> 1: ("+n.getM_id()+"_received\'=true);\n";
		}
		
		edges = m_configuration.getEdgesToNeighbors(n.getM_id(),NeighborMode.FROM);
		
		for (int i=0; i<edges.size(); i++) {
			ConfigurationEdge f = edges.get(i);
			res += "\t["+f.getM_id()+"] ("+n.getM_id()+"_received) -> 1: ("+n.getM_id()+"_received\'=false);\n";
		}

		res += "endmodule\n\n";
		return res;
	}
	
	public String getRewardsCode () {
		return getEnergyRewardsCode()+"\n"+getTrafficRewardsCode()+"\n"+getTotalEnergyRewardCode();
	}
	
	public String getEnergyRewardsCode () {
		String res="";
		for (HashMap.Entry<String,ConfigurationNode> e: m_configuration.getM_nodes().entrySet()) {
			ConfigurationNode n = e.getValue();
			res += "rewards \"energy_"+n.getM_id()+"\"\n";
			LinkedList<ConfigurationEdge> edges = m_configuration.getEdgesToNeighbors(n.getM_id(),NeighborMode.FROM);
			for (int i=0; i<edges.size(); i++) {
				ConfigurationEdge f = edges.get(i);
				double bundleEnergySent = f.getM_sent()+f.getM_processing();
				res += "\t["+f.getM_id()+"] true: "+bundleEnergySent+";\n";				
			}

			edges = m_configuration.getEdgesToNeighbors(n.getM_id(),NeighborMode.TO);
			for (int i=0; i<edges.size(); i++) {
				ConfigurationEdge f = edges.get(i);
				res += "\t["+f.getM_id()+"] true: "+f.getM_received()+";\n";				
			}		
			
			res += "\ttrue: "+ n.getM_energy_idle() + ";\n";
			
			res += "endrewards\n";
		}
		return res;
	}
	
	
	public String getTotalEnergyRewardCode () {
		String dbid = m_configuration.getDatabaseId();
		String scid = m_configuration.getComputeId();
		ConfigurationEdge dbcomp = m_configuration.getEdge(dbid+"_"+scid);
		Double energyDBCompute = dbcomp.getM_sent() + dbcomp.getM_processing() + dbcomp.getM_received();
		String 	res = "rewards \"energy\"\n";
		for (HashMap.Entry<String,ConfigurationNode> e: m_configuration.getM_nodes().entrySet()) {
			ConfigurationNode n = e.getValue();
			LinkedList<ConfigurationEdge> edges = m_configuration.getEdgesToNeighbors(n.getM_id(),NeighborMode.FROM);
			for (int i=0; i<edges.size(); i++) {
				ConfigurationEdge f = edges.get(i);
				double bundleEnergy = f.getM_sent() + f.getM_processing() + f.getM_received();
				if (m_configuration.getNode(f.getM_from()).getM_type().equals(NodeType.SENSOR) &&  // For Pattern SU, additional forward 
						m_configuration.getNode(f.getM_to()).getM_type().equals(NodeType.SENSOR)) { // message when comm between sensors
						ConfigurationEdge extraEdge = m_configuration.getEdge(n.getM_id()+"_"+dbid);
						bundleEnergy += extraEdge.getM_sent() + extraEdge.getM_processing() + extraEdge.getM_received();
				}
				if (m_configuration.getNode(f.getM_from()).getM_type().equals(NodeType.SENSOR) && 
						m_configuration.getNode(f.getM_to()).getM_type().equals(NodeType.DATABASE)) {
						if (m_configuration.getM_type().equals(PatternType.SC)){ // For SU, additional energy db-compute per 2 sensor-db messages
							bundleEnergy += energyDBCompute/2.0;
						}
						if (m_configuration.getM_type().equals(PatternType.SC) || m_configuration.getM_type().equals(PatternType.CO) ) {
							bundleEnergy += m_configuration.getEnergyDBDisplays(); // additional energy db-displays
						}
				}
				
				
				res += "\t["+f.getM_id()+"] true: "+bundleEnergy+";\n";				
			}
		}
		res += "endrewards\n";
		return res;
	}
	
	
	public String getTrafficRewardsCode () {
		String 	res = "rewards \"traffic\"\n";
		for (HashMap.Entry<String,ConfigurationNode> e: m_configuration.getM_nodes().entrySet()) {
			ConfigurationNode n = e.getValue();
			LinkedList<ConfigurationEdge> edges = m_configuration.getEdgesToNeighbors(n.getM_id(),NeighborMode.FROM);
			for (int i=0; i<edges.size(); i++) {
				ConfigurationEdge f = edges.get(i);
				Double reward =1.0;
				if (m_configuration.getNode(f.getM_from()).getM_type().equals(NodeType.SENSOR) &&  // For Pattern SU, additional forward 
						m_configuration.getNode(f.getM_to()).getM_type().equals(NodeType.SENSOR)) // message when comm between sensors
					reward+=1.0;
				if (m_configuration.getM_type().equals(PatternType.SC)) // Extra db-compute message per 2 semnsor-db messages
					reward += 0.5;
				if (m_configuration.getM_type().equals(PatternType.SC) || m_configuration.getM_type().equals(PatternType.CO) ) // Extra db-display messages
					reward +=  m_configuration.getNumberOfDisplays();
				res += "\t["+f.getM_id()+"] true: "+reward+";\n";				
			}
		}
		res += "endrewards\n";
		return res;
	}
	
	public static void main(String[] args) {
    	
    	String filename = "models/archlearner_spark_output_co.json";
		PrismGenerator pg = new PrismGenerator(filename);
       	System.out.println(pg.getModelCode(ForecastAnalyzer.extractModeSnapshot(pg.getConfiguration(), 100L)));    	
    }
}