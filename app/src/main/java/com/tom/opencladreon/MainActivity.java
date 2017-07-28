package com.tom.opencladreon;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends BaseActivity {

    private final String TAG = "MainActivity";

    Handler mClHandler ;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        HandlerThread thread = new HandlerThread("ActivityCL");
        thread.start();
        mClHandler = new Handler(thread.getLooper() );


        Button btn = (Button)findViewById(R.id.btnCVCL);
        btn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                if( ! checkPermissonIfNeed() ){
                    Log.d(TAG,"check permission first ");
                    return ;
                }else{
                    testopencvocl();
                }
            }
        });

        Button btn2 = (Button)findViewById(R.id.btnUMat);
        btn2.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
            mClHandler.post(new Runnable() {
                @Override
                public void run() {
                    if( ! checkPermissonIfNeed() ){
                        Log.d(TAG,"check permission first ");
                        return ;
                    }else {
                        openCvOclMatMul();
                    }
                }
            } );
            }
        });

        TextView testView1=(TextView)findViewById(R.id.textView1);
        TextView testView2=(TextView)findViewById(R.id.textView2);
        TextView testView3=(TextView)findViewById(R.id.textView3);
        testView1.setText(getPlatformName());
        testView2.setText(getDeviceName());
        testView3.setText(testopencl());

        ((Button)findViewById(R.id.btnCpuSobel)).setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                mClHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        if( ! checkPermissonIfNeed() ){
                            Log.d(TAG,"check permission first ");
                            return ;
                        }else {
                            nativecpusobel();
                        }
                    }
                } );
            }
        });


        ((Button)findViewById(R.id.btnGpuSobel)).setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                mClHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        if( ! checkPermissonIfNeed() ){
                            Log.d(TAG,"check permission first ");
                            return ;
                        }else {
                            nativeoclsobel();
                        }
                    }
                } );
            }
        });


    }

    @Override
    protected void onDestroy() {
        mClHandler.getLooper().quit();
        mClHandler = null;
        super.onDestroy();
    }

    public native String testopencl();
    public native String getPlatformName();
    public native String getDeviceName();
    public native void testopencvocl();
    public native void openCvOclMatMul();

    public native void nativeoclsobel();
    public native void nativecpusobel();

    public native String stringFromJNI();

    static {
        System.loadLibrary("native-lib");
    }

}
