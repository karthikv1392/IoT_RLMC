package archlearner.mc.configuration;

import archlearner.mc.configuration.ConfigurationNode.Mode;

public class SensorForecastPeriod {
	private long m_start;
	private long m_end;
	ConfigurationNode.Mode m_mode;
	
	public SensorForecastPeriod (long start, long end, ConfigurationNode.Mode mode) {
		m_start = start;
		m_end = end;
		m_mode = mode;
	}

	public long getM_start() {
		return m_start;
	}

	public void setM_start(int m_start) {
		this.m_start = m_start;
	}

	public long getM_end() {
		return m_end;
	}

	public void setM_end(int m_end) {
		this.m_end = m_end;
	}

	public ConfigurationNode.Mode getM_mode() {
		return m_mode;
	}

	public void setM_mode(ConfigurationNode.Mode m_mode) {
		this.m_mode = m_mode;
	}

	@Override
	public String toString() {
		return "SensorForecastPeriod [m_start=" + m_start + ", m_end=" + m_end + ", m_mode=" + m_mode + "]";
	}
	
	public static void main(String[] args) {
		SensorForecastPeriod sfp = new SensorForecastPeriod(0, 50, Mode.CRITICAL);
		System.out.println(sfp.toString());
	}
}
