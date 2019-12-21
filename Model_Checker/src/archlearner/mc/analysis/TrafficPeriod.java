package archlearner.mc.analysis;

public class TrafficPeriod {

	Double m_start;
	Double m_end;
	Double m_volume;
	
	
	public TrafficPeriod(Double start, Double end, Double volume) {
		m_start = start;
		m_end = end;
		m_volume = volume;
	}
	
	public Double getM_start() {
		return m_start;
	}
	public void setM_start(Double m_start) {
		this.m_start = m_start;
	}
	public Double getM_end() {
		return m_end;
	}
	public void setM_end(Double m_end) {
		this.m_end = m_end;
	}
	public Double getM_volume() {
		return m_volume;
	}
	public void setM_volume(Double m_volume) {
		this.m_volume = m_volume;
	}
	@Override
	public String toString() {
		return "TrafficPeriod [m_start=" + m_start + ", m_end=" + m_end + ", m_volume=" + m_volume + "]";
	}
		
}
