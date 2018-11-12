//
//  jhType2graph.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

class jhType2graphPanel<T> : jhPanel<T>, jhPanel_p
{
    
    func commonFirstNredraw() {
        
        jhPanelID = 1 //panel ID is matched to array id of the jhDatacenter
        maxY = 25
        
        if(jhGS.s.logLevel.contains(.network2)) {
            print("ctime in jhType2graphPanel<T> = ", (self.superScene as? jhSceneTimeLine)?.currentTime)
        }
        
        dataLayer = jhType2graphLayer<T>(self, 0, maxY)
        
        dataLayer.frame = CGRect(x: jhGS.s.jhLMarginX, y: jhGS.s.jhLMarginY, width: self.bounds.width - jhGS.s.jhLMarginX, height: self.bounds.height - jhGS.s.jhLMarginY)
        dataLayer.zPosition=1
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter2.attachObserver(observer: self)
    }
    
    override func drawDatas() {
        //        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        commonFirstNredraw()
    }
    
    override func jhRedraw() {
        
        if isFixedAxesCount {
            jhDataCenter2.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter2.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter2.mCountOfdatas_view = mAllofCountOfDatas
        commonFirstNredraw()
        
//        drawAxes()
    }
    
    /// drawBackboard calls this!
    override func drawAxes() {
        axisLayer = jhDrawAxisLayer(self, layer: 0, panelID: 0, hGuide: false, countVaxis : 4, maxY: maxY)
        
        axisLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        axisLayer.zPosition=1
        axisLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(axisLayer)
        axisLayer.setNeedsDisplay()
    }
}
