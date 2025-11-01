import os
from pypylon import pylon
import cv2
from datetime import datetime

class CameraService:
    def __init__(self,
                 width=1936,
                 height=1216,
                 offset_x=0,
                 offset_y=0,
                 pixel_format="BGR8",
                 balance_red=1.5,
                 balance_blue=1.5,
                 balance_green=0.8,
                 exposure_time=300,
                 fps=300,
                 gamma=0.9,
                 gain=18):
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                raise RuntimeError("No camera found. Please check if the camera is connected and powered on.")
            
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {str(e)}")
        self.camera.Width.SetValue(width)
        self.camera.Height.SetValue(height)
        self.camera.OffsetX.SetValue(offset_x)
        self.camera.OffsetY.SetValue(offset_y)
        self.camera.BslLightSourcePreset.SetValue("Daylight5000K")
        self.camera.ExposureAuto.SetValue("Off")
        self.camera.GainAuto.SetValue("Off")
        self.camera.BslLightSourcePresetFeatureEnable.SetValue(False)
        self.camera.PixelFormat.SetValue(pixel_format)
        self.camera.BslLightSourcePreset.SetValue("Off")
        self.camera.BalanceRatioSelector.SetValue("Red")
        self.camera.BalanceRatio.SetValue(balance_red)
        self.camera.BalanceRatioSelector.SetValue("Blue")
        self.camera.BalanceRatio.SetValue(balance_blue)
        self.camera.BalanceRatioSelector.SetValue("Green")
        self.camera.BalanceRatio.SetValue(balance_green)
        self.camera.Gain.SetValue(gain)
        self.camera.AcquisitionFrameRate.SetValue(fps)
        self.camera.BalanceWhiteAuto.SetValue("Off")
        self.camera.TriggerSelector.SetValue("FrameStart")
        self.camera.TriggerMode.SetValue("Off")
        self.camera.TriggerSource.SetValue("PeriodicSignal1")
        self.camera.ExposureTime.SetValue(exposure_time)
        self.camera.Gamma.SetValue(gamma)
        self.camera.AcquisitionMode.SetValue("Continuous")
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_frame(self):
        with self.camera.RetrieveResult(
            5000, pylon.TimeoutHandling_Return
        ) as grab:
            if grab.GrabSucceeded():
                frame = grab.GetArray()
                return frame

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()
        cv2.destroyAllWindows()