//
//  jhType3graphPanel.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 31/10/2018.
//  Copyright © 2018 hajunho.com. All rights reserved.
//

import UIKit

class jhType3graphPanel<T> : jhPanel<T> {
    override func drawDatas() {
        if(GS.s.logLevel.contains(.network2)) {
            print("ctime in jhType3graphPanel<T> = ", (self.superScene as? jhSceneTimeLine)?.currentTime)
        }
        //        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        dataLayer = jhType3graphLayer<T>(self, 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter.attachObserver(observer: self)
    }
    
    override func jhRedraw() {
        print("hjh", xDistance)
        drawAxes()
        
        if isFixedAxesCount {
            jhDataCenter.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        
        dataLayer = jhType3graphLayer(self, 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter.attachObserver(observer: self)
        
        //        drawAxes()
    }
    
    override func drawAxes() {
        axisLayer = jhDrawAxisLayer(self, layer: 0, panelID: 0, hGuide: false)
        
        axisLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        axisLayer.zPosition=1
        axisLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(axisLayer)
        axisLayer.setNeedsDisplay()
        //        jhDataCenter.attachObserver(observer: self)
    }
}
