package com.tom.opencladreon;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView testView1=(TextView)findViewById(R.id.textView1);
        TextView testView2=(TextView)findViewById(R.id.textView2);
        TextView testView3=(TextView)findViewById(R.id.textView3);
        testView1.setText(getPlatformName());
        testView2.setText(getDeviceName());
        testView3.setText(testopencl());
    }


    public native String testopencl();
    public native String getPlatformName();
    public native String getDeviceName();
    public native String stringFromJNI();

    static {
        System.loadLibrary("native-lib");
    }

}
