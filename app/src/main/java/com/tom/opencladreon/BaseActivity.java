package com.tom.opencladreon;

import android.Manifest;
import android.app.Activity;
import android.app.ActivityManager;
import android.content.pm.PackageManager;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by hl.he on 2017/6/6.
 */

public class BaseActivity extends Activity {

    private final static String TAG = "BaseActivity";

    protected Handler mMsgHandler = new Handler();
    public void toastMessage(final String message){
        mMsgHandler.post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(getActivity(), message,Toast.LENGTH_LONG).show();
            }
        });
    }

    Activity getActivity() {
        return BaseActivity.this;
    }

    @Override
    public void onLowMemory() {
        ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(info);
        String message = "meminfo  系统剩余内存:" + (info.availMem >> 10) + "k"
                + "  " + "系统是否处于低内存运行：" + info.lowMemory + "  " + "当系统剩余内存低于"
                + (info.threshold >> 10) + "k" + "时就看成低内存运行";
        toastMessage(message );
        Log.e(TAG,"onLowMemory" + message );
        super.onLowMemory();
    }


    private boolean mCameraAvailable = false ;
    private boolean mStorageAvailable = false ;
    private boolean mMicAvailable = false ;
    /*
    *   如果拒绝了,有两种情况出现
    *   1.不会再提示用户,直接onRequestPermissionsResult grantResults = PERMISSION_DENIED,这时需要到设置中授权(小米5默认/VR手机'不再询问')
    *   2.依旧再提示用户,用户需要授权即可(VR手机)
    * */
    public boolean checkPermissonIfNeed(){
        if(mCameraAvailable && mMicAvailable && mStorageAvailable){
            return true ;
        }
        ArrayList<String> pers_list = new ArrayList<String>();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            pers_list.add(Manifest.permission.CAMERA );
        }else{
            mCameraAvailable = true ;
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO )!= PackageManager.PERMISSION_GRANTED) {
            pers_list.add(Manifest.permission.RECORD_AUDIO);
        }else{
            mMicAvailable = true ;
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            pers_list.add(Manifest.permission.WRITE_EXTERNAL_STORAGE );
        }else{
            mStorageAvailable = true ;
        }
        if(pers_list.size() != 0 ){
            String[] pers_array =new String[pers_list.size()];
            pers_list.toArray(pers_array);
            Log.w(TAG,"request permission = " + Arrays.toString(pers_array) );
            ActivityCompat.requestPermissions(this, pers_array, CAMERA_REQUEST_CODE);
            return false ;
        }
        return true ;
    }


    private final int CAMERA_REQUEST_CODE = 111;
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {

        if (requestCode == CAMERA_REQUEST_CODE) {
            for(int i = 0 ; i < permissions.length ; i++){
                Log.i(TAG,String.format("Result: permissions %s grant %d ", permissions[i] , grantResults[i]) );
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED){
                    continue;
                }
                switch( permissions[i] ){
                    case Manifest.permission.CAMERA:
                        mCameraAvailable = true ;
                        break;
                    case Manifest.permission.RECORD_AUDIO:
                        mMicAvailable = true;
                        break;
                    case Manifest.permission.WRITE_EXTERNAL_STORAGE:
                        mStorageAvailable  = true ;
                        break;
                }
            }
        }
    }

}
