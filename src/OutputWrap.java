
public class OutputWrap {
	
	/***
	 * The reason this class exists is because Wrapper objects are immutable.
	 * So, having the the references of objects set in linked places, avoids
	 * the need to re-iterate several times to update the same value, as updating
	 * it once, updates it everywhere the reference is present.
	 * */

	private float output;
	
	public OutputWrap(float output) {
		this.output = output;
	}

	public float getOutput() {
		return output;
	}

	public void setOutput(float output) {
		this.output = output;
	}
}
