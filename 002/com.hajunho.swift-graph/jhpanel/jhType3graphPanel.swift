//
//  jhType3graphPanel.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 31/10/2018.
//  Copyright © 2018 hajunho.com. All rights reserved.
//

import UIKit

class jhType3graphPanel<T> : jhPanel<T>, jhPanel_p
{
    
    func commonFirstNredraw() {
        
        jhPanelID = 2 //panel ID is matched to array id of the jhDatacenter
        maxY = 15
        
        if(GS.s.logLevel.contains(.network2)) {
            print("ctime in jhType3graphPanel<T> = ", (self.superScene as? jhSceneTimeLine)?.currentTime)
        }
        
        dataLayer = jhType3graphLayer<T>(self, 0, maxY)
        
        dataLayer.frame = CGRect(x: GS.s.jhLMarginX, y: GS.s.jhLMarginY, width: self.bounds.width - GS.s.jhLMarginX, height: self.bounds.height - GS.s.jhLMarginY)
        dataLayer.zPosition=1
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter.attachObserver(observer: self)
    }
    
    override func drawDatas() {
        //        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        commonFirstNredraw()
    }
    
    override func jhRedraw() {
        //        drawAxes()
        
        if isFixedAxesCount {
            jhDataCenter.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        commonFirstNredraw()
    }
    
    /// drawBackboard calls this!
    override func drawAxes() {
        axisLayer = jhDrawAxisLayer(self, layer: 0, panelID: 0, hGuide: false, countVaxis: 2, maxY: maxY)
        
        axisLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        axisLayer.zPosition=1
        axisLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(axisLayer)
        axisLayer.setNeedsDisplay()
    }
}

