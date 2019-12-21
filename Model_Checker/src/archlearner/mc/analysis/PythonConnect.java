package archlearner.mc.analysis;


import py4j.GatewayServer;
public class PythonConnect {
	// This is the entrypoint to connect
	
	private Analyzer analyze;
	
    public PythonConnect() {
      analyze = new Analyzer(); 
     
    }

    public Analyzer getAnalyze() {
        return analyze;
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new PythonConnect());
        System.out.print(gatewayServer.getPort());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}
