package com.tom.opencladreon;

import android.os.Handler;
import android.os.HandlerThread;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class RunKernelActivity extends BaseActivity {

    private static final String TAG = "RunKernelActivity" ;



    // 底层通过 /data/data/com.tom.opencladreon/app_opencl_dir/sgemm.cl
    private void copyFile(final String f) {
        InputStream in;
        try {
            in = getAssets().open(f);
            final File of = new File(getDir("opencl_dir",MODE_PRIVATE), f);
            if( of.exists() && of.length() == in.available() ){
                Log.w(TAG,"f is Exists, Do Not Copy Again , Make sure it's correctly copied last time");
                in.close();
                return ; // 如果已经存在 请确保上一次拷贝是成功的
            }else if ( of.exists() ){ // 文件大小不一样
                Log.w(TAG,"File Size Not Correct: "
                        + String.format( "Current %d Source %d " , of.length(), in.available() )
                );
            }
            final OutputStream out = new FileOutputStream(of);

            final byte b[] = new byte[65535];
            int sz = 0;
            while ((sz = in.read(b)) > 0) {
                out.write(b, 0, sz);
            }
            in.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    Handler mClHandler ;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_run_kernel);

        HandlerThread thread = new HandlerThread("ActivityCL");
        thread.start();
        mClHandler = new Handler(thread.getLooper() );

        ((Button)findViewById(R.id.btnGEMM)).setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(final View v) {
                if( checkPermissonIfNeed() ){
                    v.setEnabled(false);
                    mClHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            copyFile("sgemm.cl");
                            boolean done = nativeRunSGEMM();
                            toastMessage( done?"Done":"Fail" );
                            mMsgHandler.post(new Runnable() {
                                @Override
                                public void run() {
                                    v.setEnabled(true);
                                }
                            });
                        }
                    } );
                }
            }
        });
    }

    @Override
    protected void onDestroy() {
        mClHandler.getLooper().quit();
        mClHandler = null;
        super.onDestroy();
    }


    /////////////////////////////////////////////////////

    public native boolean nativeRunSGEMM();
    /////////////////////////////////////////////////////

    static boolean sfoundLibrary = true;
    static {
        try {
            System.loadLibrary("native-lib");
        }
        catch (UnsatisfiedLinkError e) {
            sfoundLibrary = false;
        }
    }


}
